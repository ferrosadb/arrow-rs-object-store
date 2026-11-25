// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Human-friendly duration parsing and formatting for configuration values.
//!
//! This is a port of `humantime` 2.3.0's `parse_duration` and
//! `format_duration` (MIT OR Apache-2.0, <https://github.com/chronotope/humantime>),
//! which object_store used previously. The crate is unmaintained
//! (RUSTSEC-2026-0542), so the grammar is carried here unchanged: the same
//! inputs are accepted with the same results and errors, and durations format
//! to the same strings. The tests compare against humantime as a
//! dev-dependency oracle.
//!
//! jiff's "friendly" duration format is not a substitute: it treats months
//! and years as calendar units (humantime uses fixed 30.44 and 365.25 day
//! lengths), does not accept every humantime unit spelling, and formats with
//! different unit labels.

use std::error::Error as StdError;
use std::fmt;
use std::str::{Chars, FromStr};
use std::time::Duration;

/// Error parsing human-friendly duration
#[derive(Debug, PartialEq, Clone)]
pub(crate) enum Error {
    /// Invalid character during parsing
    ///
    /// More specifically anything that is not alphanumeric is prohibited
    ///
    /// The field is an byte offset of the character in the string.
    InvalidCharacter(usize),
    /// Non-numeric value where number is expected
    ///
    /// This usually means that either time unit is broken into words,
    /// e.g. `m sec` instead of `msec`, or just number is omitted,
    /// for example `2 hours min` instead of `2 hours 1 min`
    ///
    /// The field is an byte offset of the errorneous character
    /// in the string.
    NumberExpected(usize),
    /// Unit in the number is not one of allowed units
    ///
    /// See documentation of `parse_duration` for the list of supported
    /// time units.
    ///
    /// The two fields are start and end (exclusive) of the slice from
    /// the original string, containing errorneous value
    UnknownUnit {
        /// Start of the invalid unit inside the original string
        start: usize,
        /// End of the invalid unit inside the original string
        end: usize,
        /// The unit verbatim
        unit: String,
        /// A number associated with the unit
        value: u64,
    },
    /// The numeric value exceeds the limits of this library.
    ///
    /// This can mean two things:
    /// - The value is too large to be useful.
    ///   For instance, the maximum duration written with subseconds unit is about 3000 years.
    /// - The attempted precision is not supported.
    ///   For instance, a duration of `0.5ns` is not supported,
    ///   because durations below one nanosecond cannot be represented.
    // NOTE: it would be more logical to create a separate `NumberPrecisionLimit` error,
    // but that would be a breaking change. Reconsider this for the next major version.
    NumberOverflow,
    /// The value was an empty string (or consists only whitespace)
    Empty,
}

impl StdError for Error {}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidCharacter(offset) => write!(f, "invalid character at {}", offset),
            Self::NumberExpected(offset) => write!(f, "expected number at {}", offset),
            Self::UnknownUnit { unit, value, .. } if unit.is_empty() => {
                write!(f, "time unit needed, for example {0}sec or {0}ms", value)
            }
            Self::UnknownUnit { unit, .. } => {
                write!(
                    f,
                    "unknown time unit {:?}, \
                    supported units: ns, us/µs, ms, sec, min, hours, days, \
                    weeks, months, years (and few variations)",
                    unit
                )
            }
            Self::NumberOverflow => write!(
                f,
                "number is too large or cannot be represented without a lack of precision (values below 1ns are not supported)"
            ),
            Self::Empty => write!(f, "value was empty"),
        }
    }
}

/// A wrapper type that allows you to Display a Duration
#[derive(Debug, Clone)]
pub(crate) struct FormattedDuration(Duration);

trait OverflowOp: Sized {
    fn mul(self, other: Self) -> Result<Self, Error>;
    fn add(self, other: Self) -> Result<Self, Error>;
    fn div(self, other: Self) -> Result<Self, Error>;
}

impl OverflowOp for u64 {
    fn mul(self, other: Self) -> Result<Self, Error> {
        self.checked_mul(other).ok_or(Error::NumberOverflow)
    }
    fn add(self, other: Self) -> Result<Self, Error> {
        self.checked_add(other).ok_or(Error::NumberOverflow)
    }
    fn div(self, other: Self) -> Result<Self, Error> {
        match self % other {
            0 => Ok(self / other),
            _ => Err(Error::NumberOverflow),
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct Fraction {
    numerator: u64,
    denominator: u64,
}

struct Parser<'a> {
    iter: Chars<'a>,
    src: &'a str,
}

impl Parser<'_> {
    fn parse(mut self) -> Result<Duration, Error> {
        let mut n = self.parse_first_char()?.ok_or(Error::Empty)?; // integer part
        let mut out = Duration::ZERO;
        'outer: loop {
            let mut frac = None; // fractional part
            let mut off = self.off();
            while let Some(c) = self.iter.next() {
                match c {
                    '0'..='9' => {
                        n = n
                            .checked_mul(10)
                            .and_then(|x| x.checked_add(c as u64 - '0' as u64))
                            .ok_or(Error::NumberOverflow)?;
                    }
                    c if c.is_whitespace() => {}
                    'a'..='z' | 'A'..='Z' | 'µ' => {
                        break;
                    }
                    '.' => {
                        // decimal separator, the fractional part begins now
                        frac = Some(self.parse_fractional_part(&mut off)?);
                        break;
                    }
                    _ => {
                        return Err(Error::InvalidCharacter(off));
                    }
                }
                off = self.off();
            }
            let start = off;
            let mut off = self.off();
            while let Some(c) = self.iter.next() {
                match c {
                    '0'..='9' => {
                        self.parse_unit(n, frac, start, off, &mut out)?;
                        n = c as u64 - '0' as u64;
                        continue 'outer;
                    }
                    c if c.is_whitespace() => break,
                    'a'..='z' | 'A'..='Z' | 'µ' => {}
                    _ => {
                        return Err(Error::InvalidCharacter(off));
                    }
                }
                off = self.off();
            }

            self.parse_unit(n, frac, start, off, &mut out)?;
            n = match self.parse_first_char()? {
                Some(n) => n,
                None => return Ok(out),
            };
        }
    }

    fn parse_first_char(&mut self) -> Result<Option<u64>, Error> {
        let off = self.off();
        for c in self.iter.by_ref() {
            match c {
                '0'..='9' => {
                    return Ok(Some(c as u64 - '0' as u64));
                }
                c if c.is_whitespace() => continue,
                _ => {
                    return Err(Error::NumberExpected(off));
                }
            }
        }
        Ok(None)
    }

    fn parse_fractional_part(&mut self, off: &mut usize) -> Result<Fraction, Error> {
        let mut numerator = 0u64;
        let mut denominator = 1u64;
        let mut zeros = true;
        while let Some(c) = self.iter.next() {
            match c {
                '0' => {
                    denominator = denominator.checked_mul(10).ok_or(Error::NumberOverflow)?;
                    if !zeros {
                        numerator = numerator.checked_mul(10).ok_or(Error::NumberOverflow)?;
                    }
                }
                '1'..='9' => {
                    zeros = false;
                    denominator = denominator.checked_mul(10).ok_or(Error::NumberOverflow)?;
                    numerator = numerator
                        .checked_mul(10)
                        .and_then(|x| x.checked_add(c as u64 - '0' as u64))
                        .ok_or(Error::NumberOverflow)?;
                }
                c if c.is_whitespace() => {}
                'a'..='z' | 'A'..='Z' | 'µ' => {
                    break;
                }
                _ => {
                    return Err(Error::InvalidCharacter(*off));
                }
            };
            // update the offset used by the parsing loop
            *off = self.off();
        }
        if denominator == 1 {
            // no digits were given after the separator, e.g. "1."
            return Err(Error::InvalidCharacter(*off));
        }
        Ok(Fraction {
            numerator,
            denominator,
        })
    }

    fn off(&self) -> usize {
        self.src.len() - self.iter.as_str().len()
    }

    fn parse_unit(
        &mut self,
        n: u64,
        frac: Option<Fraction>,
        start: usize,
        end: usize,
        out: &mut Duration,
    ) -> Result<(), Error> {
        let unit = match Unit::from_str(&self.src[start..end]) {
            Ok(u) => u,
            Err(()) => {
                return Err(Error::UnknownUnit {
                    start,
                    end,
                    unit: self.src[start..end].to_owned(),
                    value: n,
                });
            }
        };

        // add the integer part
        let (sec, nsec) = match unit {
            Unit::Nanosecond => (0u64, n),
            Unit::Microsecond => (0u64, n.mul(1000)?),
            Unit::Millisecond => (0u64, n.mul(1_000_000)?),
            Unit::Second => (n, 0),
            Unit::Minute => (n.mul(60)?, 0),
            Unit::Hour => (n.mul(3600)?, 0),
            Unit::Day => (n.mul(86400)?, 0),
            Unit::Week => (n.mul(86400 * 7)?, 0),
            Unit::Month => (n.mul(2_630_016)?, 0), // 30.44d
            Unit::Year => (n.mul(31_557_600)?, 0), // 365.25d
        };
        add_current(sec, nsec, out)?;

        // add the fractional part
        if let Some(Fraction {
            numerator: n,
            denominator: d,
        }) = frac
        {
            let (sec, nsec) = match unit {
                Unit::Nanosecond => return Err(Error::NumberOverflow),
                Unit::Microsecond => (0, n.mul(1000)?.div(d)?),
                Unit::Millisecond => (0, n.mul(1_000_000)?.div(d)?),
                Unit::Second => (0, n.mul(1_000_000_000)?.div(d)?),
                Unit::Minute => (0, n.mul(60_000_000_000)?.div(d)?),
                Unit::Hour => (n.mul(3600)?.div(d)?, 0),
                Unit::Day => (n.mul(86400)?.div(d)?, 0),
                Unit::Week => (n.mul(86400 * 7)?.div(d)?, 0),
                Unit::Month => (n.mul(2_630_016)?.div(d)?, 0), // 30.44d
                Unit::Year => (n.mul(31_557_600)?.div(d)?, 0), // 365.25d
            };
            add_current(sec, nsec, out)?;
        }

        Ok(())
    }
}

fn add_current(mut sec: u64, nsec: u64, out: &mut Duration) -> Result<(), Error> {
    let mut nsec = (out.subsec_nanos() as u64).add(nsec)?;
    if nsec > 1_000_000_000 {
        sec = sec.add(nsec / 1_000_000_000)?;
        nsec %= 1_000_000_000;
    }
    sec = out.as_secs().add(sec)?;
    *out = Duration::new(sec, nsec as u32);
    Ok(())
}

enum Unit {
    Nanosecond,
    Microsecond,
    Millisecond,
    Second,
    Minute,
    Hour,
    Day,
    Week,
    Month,
    Year,
}

impl FromStr for Unit {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "nanos" | "nsec" | "ns" => Ok(Self::Nanosecond),
            "usec" | "us" | "µs" => Ok(Self::Microsecond),
            "millis" | "msec" | "ms" => Ok(Self::Millisecond),
            "seconds" | "second" | "secs" | "sec" | "s" => Ok(Self::Second),
            "minutes" | "minute" | "min" | "mins" | "m" => Ok(Self::Minute),
            "hours" | "hour" | "hr" | "hrs" | "h" => Ok(Self::Hour),
            "days" | "day" | "d" => Ok(Self::Day),
            "weeks" | "week" | "wk" | "wks" | "w" => Ok(Self::Week),
            "months" | "month" | "M" => Ok(Self::Month),
            "years" | "year" | "yr" | "yrs" | "y" => Ok(Self::Year),
            _ => Err(()),
        }
    }
}

/// Parse duration object `1hour 12min 5s`
///
/// The duration object is a concatenation of time spans. Where each time
/// span is an integer number and a suffix. Supported suffixes:
///
/// * `nsec`, `ns` -- nanoseconds
/// * `usec`, `us`, `µs` -- microseconds
/// * `msec`, `ms` -- milliseconds
/// * `seconds`, `second`, `sec`, `s`
/// * `minutes`, `minute`, `min`, `m`
/// * `hours`, `hour`, `hr`, `hrs`, `h`
/// * `days`, `day`, `d`
/// * `weeks`, `week`, `wk`, `wks`, `w`
/// * `months`, `month`, `M` -- defined as 30.44 days
/// * `years`, `year`, `yr`, `yrs`, `y` -- defined as 365.25 days
pub(crate) fn parse_duration(s: &str) -> Result<Duration, Error> {
    if s == "0" {
        return Ok(Duration::ZERO);
    }
    Parser {
        iter: s.chars(),
        src: s,
    }
    .parse()
}

/// Formats duration into a human-readable string
///
/// Note: this format is guaranteed to have same value when using
/// parse_duration, but we can change some details of the exact composition
/// of the value.
pub(crate) fn format_duration(val: Duration) -> FormattedDuration {
    FormattedDuration(val)
}

fn item_plural(
    f: &mut fmt::Formatter<'_>,
    started: &mut bool,
    name: &str,
    value: u64,
) -> fmt::Result {
    if value > 0 {
        if *started {
            f.write_str(" ")?;
        }
        write!(f, "{}{}", value, name)?;
        if value > 1 {
            f.write_str("s")?;
        }
        *started = true;
    }
    Ok(())
}
fn item(f: &mut fmt::Formatter<'_>, started: &mut bool, name: &str, value: u32) -> fmt::Result {
    if value > 0 {
        if *started {
            f.write_str(" ")?;
        }
        write!(f, "{}{}", value, name)?;
        *started = true;
    }
    Ok(())
}

impl fmt::Display for FormattedDuration {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let secs = self.0.as_secs();
        let nanos = self.0.subsec_nanos();

        if secs == 0 && nanos == 0 {
            f.write_str("0s")?;
            return Ok(());
        }

        let years = secs / 31_557_600; // 365.25d
        let ydays = secs % 31_557_600;
        let months = ydays / 2_630_016; // 30.44d
        let mdays = ydays % 2_630_016;
        let days = mdays / 86400;
        let day_secs = mdays % 86400;
        let hours = day_secs / 3600;
        let minutes = day_secs % 3600 / 60;
        let seconds = day_secs % 60;

        let millis = nanos / 1_000_000;
        let micros = nanos / 1000 % 1000;
        let nanosec = nanos % 1000;

        let started = &mut false;
        item_plural(f, started, "year", years)?;
        item_plural(f, started, "month", months)?;
        item_plural(f, started, "day", days)?;
        item(f, started, "h", hours as u32)?;
        item(f, started, "m", minutes as u32)?;
        item(f, started, "s", seconds as u32)?;
        item(f, started, "ms", millis)?;
        // humantime prints `µs` only with its non-default `mu` feature
        item(f, started, "us", micros)?;
        item(f, started, "ns", nanosec)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{format_duration, parse_duration};
    use rand::RngExt;
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use std::time::Duration;

    /// Compare one input with humantime: same value, or the same error
    fn check_parse(input: &str) {
        let ours = parse_duration(input).map_err(|e| e.to_string());
        let oracle = humantime::parse_duration(input).map_err(|e| e.to_string());
        assert_eq!(ours, oracle, "parse mismatch for {input:?}");
    }

    fn check_format(d: Duration) {
        let ours = format_duration(d).to_string();
        assert_eq!(ours, humantime::format_duration(d).to_string(), "{d:?}");
        // the formatted value parses back to the same duration
        assert_eq!(parse_duration(&ours), Ok(d), "{ours:?}");
    }

    const UNITS: &[&str] = &[
        "nanos", "nsec", "ns", "usec", "us", "µs", "millis", "msec", "ms", "seconds", "second",
        "secs", "sec", "s", "minutes", "minute", "min", "mins", "m", "hours", "hour", "hr", "hrs",
        "h", "days", "day", "d", "weeks", "week", "wk", "wks", "w", "months", "month", "M",
        "years", "year", "yr", "yrs", "y", "", "S", "H", "Sec", "x", "mss", "µ",
    ];

    const CASES: &[&str] = &[
        "0",
        "00",
        "0s",
        "60 seconds",
        "60 s",
        "60s",
        "1m 30s",
        "1h30m",
        "1h 30m 15s",
        "2h 37min",
        "32ms",
        "4.2s",
        "1.5h",
        "0.5ns",
        "1.s",
        "1..s",
        ".1s",
        ".",
        "0.000123456789s",
        "0.000000001s",
        "1 2 s",
        "1s2",
        "s",
        "",
        " ",
        "  5s  ",
        "5 s 6 m",
        "5s,6m",
        "5s-6m",
        "-5s",
        "+5s",
        "18446744073709551615s",
        "18446744073709551616s",
        "18446744073709551615ns",
        "584942417355y",
        "584942417356y",
        "1y 1M 1w 1d 1h 1m 1s 1ms 1us 1ns",
        "1µs",
        "1 µs",
        "1e3s",
        "1_000s",
        "1\t2s",
        "1\u{00a0}s",
        "10 min ",
        "1h\n30m",
        "0.1y",
        "0.0001M",
        "1.000000000001s",
        "999999999ns 1ns",
        "1.9999999995s",
    ];

    #[test]
    fn parse_matches_humantime() {
        for input in CASES {
            check_parse(input);
        }
        let mut rng = StdRng::seed_from_u64(7);
        for _ in 0..20_000 {
            let mut input = String::new();
            for _ in 0..rng.random_range(1..=4) {
                if rng.random_bool(0.3) {
                    input.push(' ');
                }
                let digits: u64 = match rng.random_range(0..4) {
                    0 => rng.random_range(0..10),
                    1 => rng.random_range(0..100_000),
                    2 => rng.random(),
                    _ => rng.random_range(0..u64::MAX / 1_000_000),
                };
                input.push_str(&digits.to_string());
                if rng.random_bool(0.2) {
                    input.push('.');
                    input.push_str(&rng.random_range(0..1_000_000u32).to_string());
                }
                if rng.random_bool(0.3) {
                    input.push(' ');
                }
                input.push_str(UNITS[rng.random_range(0..UNITS.len())]);
            }
            check_parse(&input);
            // and a random single-character corruption
            let mut chars: Vec<char> = input.chars().collect();
            let i = rng.random_range(0..chars.len());
            chars[i] = ['.', ' ', '0', 'x', '-', 'µ', '9', 's'][rng.random_range(0..8)];
            check_parse(&chars.into_iter().collect::<String>());
        }
    }

    #[test]
    fn format_matches_humantime() {
        let fixed = [
            Duration::ZERO,
            Duration::from_nanos(1),
            Duration::from_micros(1),
            Duration::from_millis(1),
            Duration::from_secs(1),
            Duration::from_secs(60),
            Duration::from_secs(90),
            Duration::from_secs(3600),
            Duration::from_secs(86_400),
            Duration::from_secs(2_630_016),
            Duration::from_secs(31_557_600),
            Duration::from_secs(2 * 31_557_600 + 2 * 2_630_016 + 2 * 86_400 + 7_384),
            Duration::new(9420, 32_001_001),
            Duration::from_secs(253_370_764_800),
        ];
        for d in fixed {
            check_format(d);
        }
        let mut rng = StdRng::seed_from_u64(11);
        for _ in 0..20_000 {
            let secs = match rng.random_range(0..3) {
                0 => rng.random_range(0..100_000),
                1 => rng.random_range(0..100_000_000),
                _ => rng.random_range(0..253_370_764_800),
            };
            let nanos = match rng.random_range(0..3) {
                0 => 0,
                1 => rng.random_range(0..1_000) * 1_000_000,
                _ => rng.random_range(0..1_000_000_000),
            };
            check_format(Duration::new(secs, nanos));
        }
    }
}
