//! lineproto: the request and reply rules of the lang.04 echo server.
//!
//! One request is one line ending in `\n` (a `\r` just before it is
//! dropped, so `nc` and `telnet` work). The first word is the command; the
//! text is everything after the first space.
//!
//! | request      | reply                                                  |
//! |--------------|--------------------------------------------------------|
//! | `PING`       | `PONG`                                                 |
//! | `ECHO <text>`| `<text>`, byte for byte                                |
//! | `LEN <text>` | the UTF-8 byte length of `<text>`, from C's `strlen`   |
//! | `QUIT`       | `BYE`, then the server closes the connection           |
//! | anything else| `ERR <reason>`; the connection stays open              |
//!
//! Nothing here does I/O. The binary in `echo/` reads lines from a socket
//! and writes `reply(line)` back, so every rule is testable without a socket.

use std::ffi::CString;
use std::fmt;
use std::os::raw::c_char;

/// A parsed request. `Echo` and `Len` own their text (a `String`), so a
/// `Command` outlives the buffer the line was read into.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Command {
    Ping,
    Echo(String),
    Len(String),
    Quit,
}

/// Every way a line can fail. The `Display` impl is the text after `ERR `.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProtoError {
    /// The line held no bytes before its `\n`.
    Empty,
    /// The line's bytes are not valid UTF-8.
    InvalidUtf8,
    /// The first word is not a command; carries that word.
    Unknown(String),
    /// `LEN` text holds a NUL byte, which a C string cannot carry.
    NulByte,
}

impl fmt::Display for ProtoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // SOLUTION-BEGIN lang.04
        match self {
            ProtoError::Empty => write!(f, "empty line"),
            ProtoError::InvalidUtf8 => write!(f, "invalid utf-8"),
            ProtoError::Unknown(word) => write!(f, "unknown command {word}"),
            ProtoError::NulByte => write!(f, "nul byte"),
        }
        // SOLUTION-END
    }
}

impl std::error::Error for ProtoError {}

// The C standard library is already linked into every Rust program, so this
// declaration is all it takes to call `strlen`. The compiler cannot check it:
// the signature must match <string.h> exactly, and every call is `unsafe`.
extern "C" {
    fn strlen(s: *const c_char) -> usize;
}

/// Parses one line (with or without its `\n` / `\r\n`) into a command.
pub fn parse(line: &str) -> Result<Command, ProtoError> {
    // SOLUTION-BEGIN lang.04
    let line = line.strip_suffix('\n').unwrap_or(line);
    let line = line.strip_suffix('\r').unwrap_or(line);
    if line.is_empty() {
        return Err(ProtoError::Empty);
    }
    let (word, text) = line.split_once(' ').unwrap_or((line, ""));
    match word {
        "PING" => Ok(Command::Ping),
        "QUIT" => Ok(Command::Quit),
        "ECHO" => Ok(Command::Echo(text.to_string())),
        "LEN" => Ok(Command::Len(text.to_string())),
        _ => Err(ProtoError::Unknown(word.to_string())),
    }
    // SOLUTION-END
}

/// The byte length of `text` as C sees it: copy it into a NUL-terminated
/// buffer and call `strlen`. A NUL inside `text` is an error, not a short
/// answer.
pub fn c_strlen(text: &str) -> Result<usize, ProtoError> {
    // SOLUTION-BEGIN lang.04
    let c = CString::new(text).map_err(|_| ProtoError::NulByte)?;
    // SAFETY: `c` is NUL-terminated and stays alive until strlen returns.
    Ok(unsafe { strlen(c.as_ptr()) })
    // SOLUTION-END
}

/// The reply text for a command (without the trailing `\n`).
pub fn respond(cmd: &Command) -> Result<String, ProtoError> {
    // SOLUTION-BEGIN lang.04
    match cmd {
        Command::Ping => Ok("PONG".to_string()),
        Command::Echo(text) => Ok(text.clone()),
        Command::Len(text) => Ok(c_strlen(text)?.to_string()),
        Command::Quit => Ok("BYE".to_string()),
    }
    // SOLUTION-END
}

/// The whole protocol for one raw line from the socket: the reply text
/// (no `\n`) and whether the server must close the connection after it.
/// Errors become `ERR <reason>` and never close the connection.
pub fn reply(line: &[u8]) -> (String, bool) {
    // SOLUTION-BEGIN lang.04
    let answer = std::str::from_utf8(line)
        .map_err(|_| ProtoError::InvalidUtf8)
        .and_then(parse)
        .and_then(|cmd| respond(&cmd).map(|text| (text, cmd == Command::Quit)));
    match answer {
        Ok(pair) => pair,
        Err(e) => (format!("ERR {e}"), false),
    }
    // SOLUTION-END
}

// Your own tests live next to the code they test. `cargo test` in the
// workspace runs them; `ss check lang.04` runs them too.
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ping_is_pong() {
        assert_eq!(reply(b"PING\n"), ("PONG".to_string(), false));
    }

    #[test]
    fn echo_keeps_everything_after_the_first_space() {
        assert_eq!(parse("ECHO  two  spaces\r\n"), Ok(Command::Echo(" two  spaces".to_string())));
    }

    #[test]
    fn len_counts_bytes_not_characters() {
        assert_eq!(c_strlen("h\u{e9}llo"), Ok(6));
    }

    #[test]
    fn errors_keep_the_connection_open() {
        assert_eq!(reply(b"JUMP high\n"), ("ERR unknown command JUMP".to_string(), false));
    }
}
