use eyre::WrapErr;
use std::io::Read;
use std::os::unix::process::CommandExt;
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

/// A UTF-8 character is at most this many bytes, so capping raw reads at
/// `truncate_limit * MAX_BYTES_PER_CHAR` still captures at least
/// `truncate_limit` complete characters. The excess is discarded before it
/// ever accumulates, bounding memory independently of the final truncation.
const MAX_BYTES_PER_CHAR: usize = 4;

/// The captured result of running a shell command via [`run_bash_command`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BashOutput {
    pub stdout: String,
    pub stderr: String,
    pub exit_code: Option<i32>,
    pub timed_out: bool,
}

fn spawn_reader<R: Read + Send + 'static>(mut pipe: R, byte_cap: usize) -> mpsc::Receiver<Vec<u8>> {
    let (sender, receiver) = mpsc::channel();
    thread::spawn(move || {
        let mut buf = Vec::new();
        let mut chunk = [0_u8; 8192];
        while buf.len() < byte_cap {
            match pipe.read(&mut chunk) {
                Ok(0) | Err(_) => break,
                Ok(read_len) => {
                    let take_len = byte_cap.saturating_sub(buf.len()).min(read_len);
                    if let Some(slice) = chunk.get(..take_len) {
                        buf.extend_from_slice(slice);
                    }
                }
            }
        }
        let _ = sender.send(buf);
    });
    receiver
}

fn spawn_waiter(mut child: Child) -> mpsc::Receiver<ExitStatus> {
    let (sender, receiver) = mpsc::channel();
    thread::spawn(move || {
        if let Ok(status) = child.wait() {
            let _ = sender.send(status);
        }
    });
    receiver
}

/// Kills every process in `pid`'s process group, not just `pid` itself.
///
/// `run_bash_command` places the spawned `bash` in its own process group, so
/// this reaches backgrounded/detached descendants (e.g. `some_cmd &`) that a
/// plain kill of the `bash` leader would leave running past the timeout.
fn kill_process_group(process_id: u32) -> eyre::Result<()> {
    let group_id = i32::try_from(process_id).wrap_err("process id does not fit in i32")?;
    let neg_group_id = group_id
        .checked_neg()
        .ok_or_else(|| eyre::eyre!("process group id negation overflowed"))?;

    // SAFETY: `neg_group_id` names a process group this function's caller
    // created for its own child via `process_group(0)`, so this only signals
    // processes we spawned.
    let result = unsafe { libc::kill(neg_group_id, libc::SIGKILL) };
    if result != 0 {
        let err = std::io::Error::last_os_error();
        if err.raw_os_error() != Some(libc::ESRCH) {
            return Err(err).wrap_err("failed to kill timed-out process group");
        }
    }
    Ok(())
}

fn truncate_chars(s: &str, limit: usize) -> String {
    s.chars().take(limit).collect()
}

/// Runs `command` under `bash -c`, enforcing `timeout` and truncating
/// captured stdout/stderr to `truncate_limit` characters.
///
/// A command that exceeds `timeout` is killed and reported via
/// [`BashOutput::timed_out`] rather than left to hang. Regardless of
/// timeout, the whole process group is killed before returning, so a
/// backgrounded/detached descendant (`some_cmd &`) can't outlive the call.
///
/// # Errors
///
/// Returns an error if the command cannot be spawned, or if killing or
/// reaping a process fails.
pub fn run_bash_command(
    command: &str,
    timeout: Duration,
    truncate_limit: usize,
) -> eyre::Result<BashOutput> {
    let mut child = Command::new("bash")
        .arg("-c")
        .arg(command)
        .process_group(0)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .wrap_err("failed to spawn bash command")?;

    let stdout_pipe = child
        .stdout
        .take()
        .ok_or_else(|| eyre::eyre!("child process has no stdout handle"))?;
    let stderr_pipe = child
        .stderr
        .take()
        .ok_or_else(|| eyre::eyre!("child process has no stderr handle"))?;

    let byte_cap = truncate_limit.saturating_mul(MAX_BYTES_PER_CHAR);
    let stdout_receiver = spawn_reader(stdout_pipe, byte_cap);
    let stderr_receiver = spawn_reader(stderr_pipe, byte_cap);

    let pid = child.id();
    let waiter = spawn_waiter(child);

    let (exit_code, timed_out) = match waiter.recv_timeout(timeout) {
        Ok(status) => (status.code(), false),
        Err(mpsc::RecvTimeoutError::Timeout) => {
            kill_process_group(pid)?;
            waiter
                .recv()
                .wrap_err("failed to reap killed child process")?;
            (None, true)
        }
        Err(mpsc::RecvTimeoutError::Disconnected) => {
            return Err(eyre::eyre!(
                "waiter thread exited without reporting a status"
            ));
        }
    };

    // The bash leader exiting doesn't mean the process group is empty: a
    // backgrounded/detached job (`some_cmd &`) can outlive it. Sweep the
    // whole group unconditionally so nothing from this call survives past
    // its return, whether the leader hit the timeout or exited on its own.
    kill_process_group(pid)?;

    let stdout_bytes = stdout_receiver.recv().unwrap_or_default();
    let stderr_bytes = stderr_receiver.recv().unwrap_or_default();

    Ok(BashOutput {
        stdout: truncate_chars(&String::from_utf8_lossy(&stdout_bytes), truncate_limit),
        stderr: truncate_chars(&String::from_utf8_lossy(&stderr_bytes), truncate_limit),
        exit_code,
        timed_out,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn captures_stdout_stderr_and_exit_status_for_a_fast_command() {
        let output = run_bash_command(
            "echo out; echo err >&2; exit 3",
            Duration::from_secs(5),
            1000,
        )
        .unwrap();

        assert_eq!(output.stdout, "out\n");
        assert_eq!(output.stderr, "err\n");
        assert_eq!(output.exit_code, Some(3));
        assert!(!output.timed_out);
    }

    #[test]
    fn kills_and_reports_timeout_for_a_command_exceeding_the_timeout() {
        let output = run_bash_command("sleep 5", Duration::from_millis(50), 1000).unwrap();

        assert!(output.timed_out);
        assert_eq!(output.exit_code, None);
    }

    #[test]
    fn kills_backgrounded_descendant_processes_too() {
        let marker = std::env::temp_dir().join(format!(
            "tib-bash-tool-test-{}-{}",
            std::process::id(),
            line!()
        ));
        let marker_path = marker.to_str().unwrap();

        run_bash_command(
            &format!("(sleep 2 && touch {marker_path}) &"),
            Duration::from_millis(100),
            1000,
        )
        .unwrap();

        thread::sleep(Duration::from_secs(3));

        assert!(!marker.exists());
        let _ = std::fs::remove_file(&marker);
    }

    #[test]
    fn truncates_captured_output_past_the_character_limit() {
        let output = run_bash_command("printf '%050d' 0", Duration::from_secs(5), 10).unwrap();

        assert_eq!(output.stdout.chars().count(), 10);
        assert_eq!(output.stdout, "0000000000");
        assert!(!output.timed_out);
    }
}
