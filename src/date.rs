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

//! Date parsing and formatting on top of [`jiff`].
//!
//! object_store previously used `chrono` for every date it parsed or printed.
//! The parsers here reproduce the exact grammar of the chrono 0.4.39 routines
//! they replace (accepted inputs, resulting instant, and error kind), so that
//! responses from S3, Azure, GCS and WebDAV servers are interpreted exactly as
//! before. The tests compare against chrono as a dev-dependency oracle.
//!
//! The ported grammar is derived from chrono (MIT/Apache-2.0),
//! `src/format/{parse,scan,parsed}.rs`.
//!
//! Two deliberate differences, both because [`Timestamp`] cannot represent
//! the value:
//!
//! * Instants outside jiff's range (`-009999-01-02T01:59:59Z` to
//!   `9999-12-30T22:00:00.999999999Z`) are rejected with
//!   [`DateParseErrorKind::OutOfRange`]; chrono accepted years up to ±262143.
//! * A leap second (`:60`) is accepted where chrono accepted it, and is
//!   represented as second `:59` with the parsed sub-second nanoseconds, which
//!   is how jiff's own parsers treat leap seconds.

#![allow(dead_code)] // each parser/formatter is only used by some feature combinations

use std::fmt;

use jiff::Timestamp;
use jiff::civil::{Date, DateTime, Weekday};
use jiff::tz::Offset;

/// The reason a date string was rejected, mirroring `chrono::format::ParseErrorKind`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DateParseErrorKind {
    /// A field or the resulting instant is out of range
    OutOfRange,
    /// The fields are inconsistent (for example a weekday that does not match the date)
    Impossible,
    /// Required fields are missing
    NotEnough,
    /// An unexpected character was found
    Invalid,
    /// The input ended early
    TooShort,
    /// The input has trailing characters
    TooLong,
}

/// A date string could not be parsed
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DateParseError {
    kind: DateParseErrorKind,
    /// The fields were valid for chrono but the instant is outside jiff's range
    beyond_jiff_range: bool,
}

impl DateParseError {
    const fn new(kind: DateParseErrorKind) -> Self {
        Self {
            kind,
            beyond_jiff_range: false,
        }
    }

    pub(crate) fn kind(&self) -> DateParseErrorKind {
        self.kind
    }
}

impl fmt::Display for DateParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Same messages as chrono's `ParseError`
        f.write_str(match self.kind {
            DateParseErrorKind::OutOfRange => "input is out of range",
            DateParseErrorKind::Impossible => "no possible date and time matching input",
            DateParseErrorKind::NotEnough => "input is not enough for unique date and time",
            DateParseErrorKind::Invalid => "input contains invalid characters",
            DateParseErrorKind::TooShort => "premature end of input",
            DateParseErrorKind::TooLong => "trailing input",
        })
    }
}

impl std::error::Error for DateParseError {}

type ParseResult<T> = Result<T, DateParseError>;

const OUT_OF_RANGE: DateParseError = DateParseError::new(DateParseErrorKind::OutOfRange);
const IMPOSSIBLE: DateParseError = DateParseError::new(DateParseErrorKind::Impossible);
const NOT_ENOUGH: DateParseError = DateParseError::new(DateParseErrorKind::NotEnough);
const INVALID: DateParseError = DateParseError::new(DateParseErrorKind::Invalid);
const TOO_SHORT: DateParseError = DateParseError::new(DateParseErrorKind::TooShort);
const TOO_LONG: DateParseError = DateParseError::new(DateParseErrorKind::TooLong);
/// chrono could represent the value but jiff cannot
const BEYOND_JIFF_RANGE: DateParseError = DateParseError {
    kind: DateParseErrorKind::OutOfRange,
    beyond_jiff_range: true,
};

/// Parses an RFC 2822 date, as used by the HTTP `Last-Modified` header.
///
/// Equivalent to `chrono::DateTime::parse_from_rfc2822(s)?.with_timezone(&Utc)`.
pub(crate) fn parse_rfc2822(s: &str) -> ParseResult<Timestamp> {
    let mut parsed = Parsed::default();
    let rest = rfc2822(&mut parsed, s)?;
    if !rest.is_empty() {
        return Err(TOO_LONG);
    }
    parsed.to_timestamp()
}

/// Parses an RFC 3339 date, as found in S3 / GCS list responses and AWS credentials.
///
/// Equivalent to chrono's `FromStr for DateTime<FixedOffset>` followed by
/// `with_timezone(&Utc)`, which is what chrono's serde `Deserialize` impl uses.
/// This accepts chrono's relaxed RFC 3339 form: unpadded fields, a `T`, `t` or
/// space separator, spaces between components, `UTC` as an offset, and an
/// offset without a colon.
pub(crate) fn parse_rfc3339(s: &str) -> ParseResult<Timestamp> {
    let mut parsed = Parsed::default();
    let rest = rfc3339_relaxed(&mut parsed, s)?;
    if !rest.trim_start().is_empty() {
        return Err(TOO_LONG);
    }
    parsed.to_timestamp()
}

/// Parses an RFC 1123 date such as `Sun, 06 Nov 1994 08:49:37 GMT`.
///
/// Equivalent to `chrono::NaiveDateTime::parse_from_str(s, "%a, %d %h %Y %T GMT")`
/// interpreted as UTC.
pub(crate) fn parse_rfc1123(s: &str) -> ParseResult<Timestamp> {
    let mut parsed = Parsed::default();
    let mut s = s;
    let (rest, weekday) = short_weekday(s)?;
    parsed.weekday = Some(weekday);
    s = literal(rest, ",")?.trim_start();
    s = numeric(s, 2, &mut parsed.day, set_day)?.trim_start();
    let (rest, month0) = short_month0(s)?;
    parsed.month = Some(i64::from(month0) + 1);
    s = rest.trim_start();
    s = signed_year(s, &mut parsed)?;
    s = s.trim_start();
    s = hms(s, &mut parsed)?.trim_start();
    s = literal(s, "GMT")?;
    if !s.is_empty() {
        return Err(TOO_LONG);
    }
    parsed.offset = Some(0);
    parsed.to_timestamp()
}

/// Parses the Azure CLI `expiresOn` format, `%Y-%m-%d %H:%M:%S.%6f`, as a civil time.
///
/// Equivalent to `chrono::NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S.%6f")`.
pub(crate) fn parse_az_cli_datetime(s: &str) -> ParseResult<DateTime> {
    let mut parsed = Parsed::default();
    let mut s = signed_year(s, &mut parsed)?;
    s = literal(s, "-")?;
    s = numeric(s, 2, &mut parsed.month, set_month)?;
    s = literal(s, "-")?;
    s = numeric(s, 2, &mut parsed.day, set_day)?;
    s = hms(s.trim_start(), &mut parsed)?;
    s = literal(s, ".")?;
    if s.len() < 6 {
        return Err(TOO_SHORT);
    }
    let (rest, v) = number(s, 6, 6)?;
    parsed.nanosecond = Some(set_nanosecond(v * 1_000)?);
    if !rest.is_empty() {
        return Err(TOO_LONG);
    }
    parsed.to_datetime()
}

/// `%H:%M:%S` as parsed by chrono's strftime parser (each number may be
/// preceded by whitespace)
fn hms<'a>(s: &'a str, parsed: &mut Parsed) -> ParseResult<&'a str> {
    let mut s = numeric(s, 2, &mut parsed.hour, set_hour)?;
    s = literal(s, ":")?;
    s = numeric(s, 2, &mut parsed.minute, set_minute)?;
    s = literal(s, ":")?;
    numeric(s, 2, &mut parsed.second, set_second)
}

/// chrono's `Numeric::Year`: optional sign with unbounded digits, else up to 4 digits
fn signed_year<'a>(s: &'a str, parsed: &mut Parsed) -> ParseResult<&'a str> {
    let s = s.trim_start();
    let (rest, v) = if let Some(digits) = s.strip_prefix('-') {
        let (rest, v) = number(digits, 1, usize::MAX)?;
        (rest, 0i64.checked_sub(v).ok_or(OUT_OF_RANGE)?)
    } else if let Some(digits) = s.strip_prefix('+') {
        number(digits, 1, usize::MAX)?
    } else {
        number(s, 1, 4)?
    };
    parsed.year = Some(set_year(v)?);
    Ok(rest)
}

/// An unsigned chrono numeric item: leading whitespace, then 1..=width digits
fn numeric<'a>(
    s: &'a str,
    width: usize,
    field: &mut Option<i64>,
    set: fn(i64) -> ParseResult<i64>,
) -> ParseResult<&'a str> {
    let (rest, v) = number(s.trim_start(), 1, width)?;
    *field = Some(set(v)?);
    Ok(rest)
}

fn literal<'a>(s: &'a str, prefix: &str) -> ParseResult<&'a str> {
    if s.len() < prefix.len() {
        return Err(TOO_SHORT);
    }
    s.strip_prefix(prefix).ok_or(INVALID)
}

/// RFC 2822 grammar, ported from chrono's `parse_rfc2822`
fn rfc2822<'a>(parsed: &mut Parsed, s: &'a str) -> ParseResult<&'a str> {
    let mut s = s.trim_start();

    if let Ok((rest, weekday)) = short_weekday(s) {
        s = rest.strip_prefix(',').ok_or(INVALID)?;
        parsed.weekday = Some(weekday);
    }

    s = s.trim_start();
    let (rest, day) = number(s, 1, 2)?;
    parsed.day = Some(set_day(day)?);
    s = space(rest)?;
    let (rest, month0) = short_month0(s)?;
    parsed.month = Some(i64::from(month0) + 1);
    s = space(rest)?;

    // two- and three-digit years are offset from 1900 / 2000
    let (rest, mut year) = number(s, 2, usize::MAX)?;
    match (s.len() - rest.len(), year) {
        (2, 0..=49) => year += 2000,
        (2, 50..=99) | (3, _) => year += 1900,
        _ => {}
    }
    parsed.year = Some(set_year(year)?);

    s = space(rest)?;
    let (rest, hour) = number(s, 2, 2)?;
    parsed.hour = Some(set_hour(hour)?);
    s = char(rest.trim_start(), b':')?.trim_start();
    let (rest, minute) = number(s, 2, 2)?;
    parsed.minute = Some(set_minute(minute)?);
    s = rest;
    if let Ok(after_colon) = char(s.trim_start(), b':') {
        let (rest, second) = number(after_colon, 2, 2)?;
        parsed.second = Some(set_second(second)?);
        s = rest;
    }

    s = space(s)?;
    let (rest, offset) = timezone_offset_2822(s)?;
    parsed.offset = Some(set_offset(i64::from(offset))?);
    s = rest;

    while let Ok(rest) = comment_2822(s) {
        s = rest;
    }
    Ok(s)
}

/// chrono's relaxed RFC 3339 grammar (`parse_rfc3339_relaxed`)
fn rfc3339_relaxed<'a>(parsed: &mut Parsed, s: &'a str) -> ParseResult<&'a str> {
    let mut s = signed_year(s, parsed)?;
    s = literal(s.trim_start(), "-")?;
    s = numeric(s, 2, &mut parsed.month, set_month)?;
    s = literal(s.trim_start(), "-")?;
    s = numeric(s, 2, &mut parsed.day, set_day)?;

    s = match s.as_bytes().first() {
        Some(b't' | b'T' | b' ') => &s[1..],
        Some(_) => return Err(INVALID),
        None => return Err(TOO_SHORT),
    };

    s = numeric(s, 2, &mut parsed.hour, set_hour)?;
    s = literal(s.trim_start(), ":")?;
    s = numeric(s, 2, &mut parsed.minute, set_minute)?;
    s = literal(s.trim_start(), ":")?;
    s = numeric(s, 2, &mut parsed.second, set_second)?;
    if let Some(frac) = s.strip_prefix('.') {
        let (rest, nanos) = nanosecond(frac)?;
        parsed.nanosecond = Some(set_nanosecond(nanos)?);
        s = rest;
    }
    s = s.trim_start();

    let (s, offset) = if s.len() >= 3 && s.as_bytes()[..3].eq_ignore_ascii_case(b"UTC") {
        (&s[3..], 0)
    } else {
        timezone_offset(s, colon_or_space, true)?
    };
    parsed.offset = Some(set_offset(i64::from(offset))?);
    Ok(s)
}

/// The fields collected while parsing, as in chrono's `Parsed`.
///
/// None of the grammars here sets a field twice, so chrono's consistency
/// checks between repeated fields are not needed.
#[derive(Debug, Default)]
struct Parsed {
    year: Option<i64>,
    month: Option<i64>,
    day: Option<i64>,
    weekday: Option<Weekday>,
    hour: Option<i64>,
    minute: Option<i64>,
    second: Option<i64>,
    nanosecond: Option<i64>,
    offset: Option<i64>,
}

impl Parsed {
    /// chrono's `to_naive_date`, restricted to year/month/day/weekday
    fn to_date(&self) -> ParseResult<Date> {
        let (Some(year), Some(month), Some(day)) = (self.year, self.month, self.day) else {
            return Err(NOT_ENOUGH);
        };
        // The setters guarantee month in 1..=12, day in 1..=31 and year within i32.
        // chrono accepts years up to ±262143; jiff only -9999..=9999.
        let year = match i16::try_from(year) {
            Ok(y) if (-9999..=9999).contains(&y) => y,
            _ if (-262_144..=262_143).contains(&year) => return Err(BEYOND_JIFF_RANGE),
            _ => return Err(OUT_OF_RANGE),
        };
        let date = Date::new(year, month as i8, day as i8).map_err(|_| OUT_OF_RANGE)?;
        match self.weekday {
            Some(weekday) if weekday != date.weekday() => Err(IMPOSSIBLE),
            _ => Ok(date),
        }
    }

    /// chrono's `to_naive_time`, with a leap second clamped to `:59`
    fn to_time(&self) -> ParseResult<(i8, i8, i8, i32)> {
        let (Some(hour), Some(minute)) = (self.hour, self.minute) else {
            return Err(NOT_ENOUGH);
        };
        let second = match self.second.unwrap_or(0) {
            v @ 0..=59 => v,
            60 => 59,
            _ => return Err(OUT_OF_RANGE),
        };
        let nano = match self.nanosecond {
            Some(v @ 0..=999_999_999) if self.second.is_some() => v,
            Some(0..=999_999_999) => return Err(NOT_ENOUGH),
            Some(_) => return Err(OUT_OF_RANGE),
            None => 0,
        };
        Ok((hour as i8, minute as i8, second as i8, nano as i32))
    }

    fn to_datetime(&self) -> ParseResult<DateTime> {
        // chrono reports the date error first, then the time error
        let date = self.to_date()?;
        let (h, m, s, ns) = self.to_time()?;
        Ok(date.at(h, m, s, ns))
    }

    /// chrono's `to_datetime` followed by conversion to UTC
    fn to_timestamp(&self) -> ParseResult<Timestamp> {
        let offset = self.offset.ok_or(NOT_ENOUGH)?;
        let datetime = self.to_datetime()?;
        // chrono's `FixedOffset::east_opt` only accepts offsets strictly within a day
        if !(-86_400 < offset && offset < 86_400) {
            return Err(OUT_OF_RANGE);
        }
        let offset = Offset::from_seconds(offset as i32).map_err(|_| OUT_OF_RANGE)?;
        offset.to_timestamp(datetime).map_err(|_| BEYOND_JIFF_RANGE)
    }
}

fn set_year(v: i64) -> ParseResult<i64> {
    i32::try_from(v).map(i64::from).map_err(|_| OUT_OF_RANGE)
}

fn set_month(v: i64) -> ParseResult<i64> {
    in_range(v, 1, 12)
}

fn set_day(v: i64) -> ParseResult<i64> {
    in_range(v, 1, 31)
}

fn set_hour(v: i64) -> ParseResult<i64> {
    in_range(v, 0, 23)
}

fn set_minute(v: i64) -> ParseResult<i64> {
    in_range(v, 0, 59)
}

fn set_second(v: i64) -> ParseResult<i64> {
    in_range(v, 0, 60)
}

fn set_nanosecond(v: i64) -> ParseResult<i64> {
    in_range(v, 0, 999_999_999)
}

fn set_offset(v: i64) -> ParseResult<i64> {
    i32::try_from(v).map(i64::from).map_err(|_| OUT_OF_RANGE)
}

fn in_range(v: i64, min: i64, max: i64) -> ParseResult<i64> {
    if (min..=max).contains(&v) {
        Ok(v)
    } else {
        Err(OUT_OF_RANGE)
    }
}

/// chrono's `scan::number`: between `min` and `max` ASCII digits
fn number(s: &str, min: usize, max: usize) -> ParseResult<(&str, i64)> {
    debug_assert!(min <= max);
    let bytes = s.as_bytes();
    if bytes.len() < min {
        return Err(TOO_SHORT);
    }
    let mut n = 0i64;
    for (i, c) in bytes.iter().take(max).copied().enumerate() {
        if !c.is_ascii_digit() {
            return if i < min {
                Err(INVALID)
            } else {
                Ok((&s[i..], n))
            };
        }
        n = n
            .checked_mul(10)
            .and_then(|n| n.checked_add(i64::from(c - b'0')))
            .ok_or(OUT_OF_RANGE)?;
    }
    Ok((&s[max.min(bytes.len())..], n))
}

/// chrono's `scan::nanosecond`: 1 to 9 significant digits, extra digits ignored
fn nanosecond(s: &str) -> ParseResult<(&str, i64)> {
    const SCALE: [i64; 10] = [
        0,
        100_000_000,
        10_000_000,
        1_000_000,
        100_000,
        10_000,
        1_000,
        100,
        10,
        1,
    ];
    let (rest, v) = number(s, 1, 9)?;
    let consumed = s.len() - rest.len();
    let v = v.checked_mul(SCALE[consumed]).ok_or(OUT_OF_RANGE)?;
    Ok((rest.trim_start_matches(|c: char| c.is_ascii_digit()), v))
}

fn short_month0(s: &str) -> ParseResult<(&str, u8)> {
    if s.len() < 3 {
        return Err(TOO_SHORT);
    }
    let b = s.as_bytes();
    let month0 = match (b[0] | 32, b[1] | 32, b[2] | 32) {
        (b'j', b'a', b'n') => 0,
        (b'f', b'e', b'b') => 1,
        (b'm', b'a', b'r') => 2,
        (b'a', b'p', b'r') => 3,
        (b'm', b'a', b'y') => 4,
        (b'j', b'u', b'n') => 5,
        (b'j', b'u', b'l') => 6,
        (b'a', b'u', b'g') => 7,
        (b's', b'e', b'p') => 8,
        (b'o', b'c', b't') => 9,
        (b'n', b'o', b'v') => 10,
        (b'd', b'e', b'c') => 11,
        _ => return Err(INVALID),
    };
    Ok((&s[3..], month0))
}

fn short_weekday(s: &str) -> ParseResult<(&str, Weekday)> {
    if s.len() < 3 {
        return Err(TOO_SHORT);
    }
    let b = s.as_bytes();
    let weekday = match (b[0] | 32, b[1] | 32, b[2] | 32) {
        (b'm', b'o', b'n') => Weekday::Monday,
        (b't', b'u', b'e') => Weekday::Tuesday,
        (b'w', b'e', b'd') => Weekday::Wednesday,
        (b't', b'h', b'u') => Weekday::Thursday,
        (b'f', b'r', b'i') => Weekday::Friday,
        (b's', b'a', b't') => Weekday::Saturday,
        (b's', b'u', b'n') => Weekday::Sunday,
        _ => return Err(INVALID),
    };
    Ok((&s[3..], weekday))
}

fn char(s: &str, c: u8) -> ParseResult<&str> {
    match s.as_bytes().first() {
        Some(&b) if b == c => Ok(&s[1..]),
        Some(_) => Err(INVALID),
        None => Err(TOO_SHORT),
    }
}

/// Mandatory whitespace
fn space(s: &str) -> ParseResult<&str> {
    let trimmed = s.trim_start();
    if trimmed.len() < s.len() {
        Ok(trimmed)
    } else if s.is_empty() {
        Err(TOO_SHORT)
    } else {
        Err(INVALID)
    }
}

fn colon_or_space(s: &str) -> ParseResult<&str> {
    Ok(s.trim_start_matches(|c: char| c == ':' || c.is_whitespace()))
}

/// chrono's `scan::timezone_offset` with `allow_missing_minutes = false`
fn timezone_offset(
    mut s: &str,
    consume_colon: fn(&str) -> ParseResult<&str>,
    allow_tz_minus_sign_and_zulu: bool,
) -> ParseResult<(&str, i32)> {
    if allow_tz_minus_sign_and_zulu {
        if let Some(b'Z' | b'z') = s.as_bytes().first() {
            return Ok((&s[1..], 0));
        }
    }
    let negative = match s.chars().next() {
        Some('+') => false,
        Some('-') => true,
        Some('\u{2212}') if allow_tz_minus_sign_and_zulu => true,
        Some(_) => return Err(INVALID),
        None => return Err(TOO_SHORT),
    };
    s = &s[s.chars().next().map_or(0, char::len_utf8)..];

    let hours = match two_bytes(s)? {
        (h1 @ b'0'..=b'9', h2 @ b'0'..=b'9') => i32::from((h1 - b'0') * 10 + (h2 - b'0')),
        _ => return Err(INVALID),
    };
    s = consume_colon(&s[2..])?;

    let minutes = match two_bytes(s)? {
        (m1 @ b'0'..=b'5', m2 @ b'0'..=b'9') => i32::from((m1 - b'0') * 10 + (m2 - b'0')),
        (b'6'..=b'9', b'0'..=b'9') => return Err(OUT_OF_RANGE),
        _ => return Err(INVALID),
    };
    s = &s[2..];

    let seconds = hours * 3600 + minutes * 60;
    Ok((s, if negative { -seconds } else { seconds }))
}

fn two_bytes(s: &str) -> ParseResult<(u8, u8)> {
    match s.as_bytes() {
        [a, b, ..] => Ok((*a, *b)),
        _ => Err(TOO_SHORT),
    }
}

fn timezone_offset_2822(s: &str) -> ParseResult<(&str, i32)> {
    let upto = s
        .as_bytes()
        .iter()
        .position(|c| !c.is_ascii_alphabetic())
        .unwrap_or(s.len());
    if upto == 0 {
        // chrono: `timezone_offset(s, |s| Ok(s), false, false, false)`
        return timezone_offset(s, |s| Ok(s), false);
    }
    let name = &s.as_bytes()[..upto];
    let rest = &s[upto..];
    let is = |n: &[u8]| name.eq_ignore_ascii_case(n);
    let hours = if is(b"gmt") || is(b"ut") || is(b"z") {
        0
    } else if is(b"edt") {
        -4
    } else if is(b"est") || is(b"cdt") {
        -5
    } else if is(b"cst") || is(b"mdt") {
        -6
    } else if is(b"mst") || is(b"pdt") {
        -7
    } else if is(b"pst") {
        -8
    } else if let [b'a'..=b'i' | b'k'..=b'y' | b'A'..=b'I' | b'K'..=b'Y'] = name {
        // military zones are treated as -0000
        0
    } else {
        return Err(INVALID);
    };
    Ok((rest, hours * 3600))
}

/// An RFC 2822 comment, with nesting and backslash escapes
fn comment_2822(s: &str) -> ParseResult<&str> {
    let s = s.trim_start();
    let mut depth = 0usize;
    let mut escaped = false;
    for (i, c) in s.bytes().enumerate() {
        if depth == 0 {
            if c != b'(' {
                return Err(INVALID);
            }
            depth = 1;
        } else if escaped {
            escaped = false;
        } else if c == b'\\' {
            escaped = true;
        } else if c == b'(' {
            depth += 1;
        } else if c == b')' {
            if depth == 1 {
                return Ok(&s[i + 1..]);
            }
            depth -= 1;
        }
    }
    Err(TOO_SHORT)
}

/// `ts + d`, or `None` if the result is outside jiff's range.
///
/// chrono's `DateTime + std::time::Duration` panicked on overflow instead.
pub(crate) fn checked_add_std(ts: Timestamp, d: std::time::Duration) -> Option<Timestamp> {
    let d = jiff::SignedDuration::try_from(d).ok()?;
    ts.checked_add(d).ok()
}

/// The civil date and time of `ts` in UTC
fn utc(ts: Timestamp) -> DateTime {
    Offset::UTC.to_datetime(ts)
}

/// A year as chrono's `%Y` prints it: 4 digits, with a sign outside 0..=9999
fn write_year(f: &mut fmt::Formatter<'_>, year: i16) -> fmt::Result {
    if (0..=9999).contains(&year) {
        write!(f, "{year:04}")
    } else {
        write!(f, "{year:+05}")
    }
}

/// Lazily formats a [`Timestamp`] in one of the fixed layouts below
#[derive(Debug, Clone, Copy)]
pub(crate) struct Formatted {
    ts: Timestamp,
    layout: Layout,
}

#[derive(Debug, Clone, Copy)]
enum Layout {
    BasicDateTime,
    BasicDate,
    Http,
    Rfc3339Secs,
}

/// `%Y%m%dT%H%M%SZ`, as used by AWS SigV4 and GCS V4 signing
pub(crate) fn basic_datetime(ts: Timestamp) -> Formatted {
    Formatted {
        ts,
        layout: Layout::BasicDateTime,
    }
}

/// `%Y%m%d`, as used in signing scopes
pub(crate) fn basic_date(ts: Timestamp) -> Formatted {
    Formatted {
        ts,
        layout: Layout::BasicDate,
    }
}

/// `%a, %d %b %Y %H:%M:%S GMT`, the HTTP-date (RFC 1123) format
pub(crate) fn http_date(ts: Timestamp) -> Formatted {
    Formatted {
        ts,
        layout: Layout::Http,
    }
}

/// RFC 3339 with whole seconds and a `Z` suffix, as chrono's
/// `to_rfc3339_opts(SecondsFormat::Secs, true)`
pub(crate) fn rfc3339_secs(ts: Timestamp) -> Formatted {
    Formatted {
        ts,
        layout: Layout::Rfc3339Secs,
    }
}

const WEEKDAYS: [&str; 7] = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
const MONTHS: [&str; 12] = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
];

impl fmt::Display for Formatted {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let dt = utc(self.ts);
        let (mo, d, h, mi, s) = (dt.month(), dt.day(), dt.hour(), dt.minute(), dt.second());
        match self.layout {
            Layout::BasicDateTime => {
                write_year(f, dt.year())?;
                write!(f, "{mo:02}{d:02}T{h:02}{mi:02}{s:02}Z")
            }
            Layout::BasicDate => {
                write_year(f, dt.year())?;
                write!(f, "{mo:02}{d:02}")
            }
            Layout::Http => {
                let weekday = WEEKDAYS[dt.weekday().to_monday_zero_offset() as usize];
                let month = MONTHS[(mo - 1) as usize];
                write!(f, "{weekday}, {d:02} {month} ")?;
                write_year(f, dt.year())?;
                write!(f, " {h:02}:{mi:02}:{s:02} GMT")
            }
            Layout::Rfc3339Secs => {
                write_year(f, dt.year())?;
                write!(f, "-{mo:02}-{d:02}T{h:02}:{mi:02}:{s:02}Z")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{NaiveDateTime, SecondsFormat, TimeZone, Utc};
    use rand::RngExt;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    type Oracle = fn(&str) -> Result<(i64, u32), chrono::format::ParseError>;
    type Ours = fn(&str) -> ParseResult<Timestamp>;

    fn chrono_rfc2822(s: &str) -> Result<(i64, u32), chrono::format::ParseError> {
        let dt = chrono::DateTime::parse_from_rfc2822(s)?.with_timezone(&Utc);
        Ok((dt.timestamp(), dt.timestamp_subsec_nanos()))
    }

    fn chrono_rfc3339(s: &str) -> Result<(i64, u32), chrono::format::ParseError> {
        let dt = s.parse::<chrono::DateTime<chrono::FixedOffset>>()?;
        let dt = dt.with_timezone(&Utc);
        Ok((dt.timestamp(), dt.timestamp_subsec_nanos()))
    }

    fn chrono_rfc1123(s: &str) -> Result<(i64, u32), chrono::format::ParseError> {
        let naive = NaiveDateTime::parse_from_str(s, "%a, %d %h %Y %T GMT")?;
        let dt = Utc.from_utc_datetime(&naive);
        Ok((dt.timestamp(), dt.timestamp_subsec_nanos()))
    }

    fn chrono_az_cli(s: &str) -> Result<(i64, u32), chrono::format::ParseError> {
        let naive = NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S.%6f")?;
        let dt = naive.and_utc();
        Ok((dt.timestamp(), dt.timestamp_subsec_nanos()))
    }

    fn ours_az_cli(s: &str) -> ParseResult<Timestamp> {
        parse_az_cli_datetime(s)
            .and_then(|dt| Offset::UTC.to_timestamp(dt).map_err(|_| BEYOND_JIFF_RANGE))
    }

    fn map_kind(e: &chrono::format::ParseError) -> DateParseErrorKind {
        use chrono::format::ParseErrorKind as K;
        match e.kind() {
            K::OutOfRange => DateParseErrorKind::OutOfRange,
            K::Impossible => DateParseErrorKind::Impossible,
            K::NotEnough => DateParseErrorKind::NotEnough,
            K::Invalid => DateParseErrorKind::Invalid,
            K::TooShort => DateParseErrorKind::TooShort,
            K::TooLong => DateParseErrorKind::TooLong,
            other => panic!("unexpected chrono error kind {other:?}"),
        }
    }

    /// Asserts that our parser and the chrono oracle agree on `input`.
    ///
    /// Returns false (without asserting) for the documented deviations: a value
    /// chrono accepts but which is outside jiff's range.
    fn check(input: &str, ours: Ours, oracle: Oracle) -> bool {
        let expected = oracle(input);
        let actual = ours(input);
        match (&expected, &actual) {
            (Ok((secs, nanos)), Ok(ts)) => {
                // chrono represents a leap second as :59 with nanos >= 1e9
                let nanos = i128::from(*nanos % 1_000_000_000);
                let expected = i128::from(*secs) * 1_000_000_000 + nanos;
                assert_eq!(ts.as_nanosecond(), expected, "value mismatch for {input:?}");
                true
            }
            (Ok((secs, _)), Err(e)) if e.beyond_jiff_range => {
                assert!(
                    Timestamp::from_second(*secs).is_err(),
                    "{input:?}: chrono accepted an in-range value we rejected"
                );
                false
            }
            (Err(_), Err(a)) if a.beyond_jiff_range => false,
            (Err(e), Err(a)) => {
                assert_eq!(map_kind(e), a.kind(), "error kind mismatch for {input:?}");
                assert_eq!(e.to_string(), a.to_string());
                true
            }
            _ => panic!("{input:?}: chrono {expected:?}, ours {actual:?}"),
        }
    }

    const RFC2822_CASES: &[&str] = &[
        "Tue, 1 Jul 2003 10:52:37 +0200",
        "Tue, 01 Jul 2003 10:52:37 +0200",
        "Wed, 18 Feb 2015 23:16:09 GMT",
        "Wed, 18 Feb 2015 23:16:09 gmt",
        "Wed, 18 Feb 2015 23:16:09 UT",
        "Wed, 18 Feb 2015 23:16:09 Z",
        "Wed, 18 Feb 2015 23:16:09 EST",
        "Wed, 18 Feb 2015 23:16:09 edt",
        "Wed, 18 Feb 2015 23:16:09 CST",
        "Wed, 18 Feb 2015 23:16:09 CDT",
        "Wed, 18 Feb 2015 23:16:09 MST",
        "Wed, 18 Feb 2015 23:16:09 MDT",
        "Wed, 18 Feb 2015 23:16:09 PST",
        "Wed, 18 Feb 2015 23:16:09 PDT",
        "Wed, 18 Feb 2015 23:16:09 A",
        "Wed, 18 Feb 2015 23:16:09 J",
        "Wed, 18 Feb 2015 23:16:09 y",
        "Wed, 18 Feb 2015 23:16:09 XYZ",
        "Wed, 18 Feb 2015 23:16:09 +0000",
        "Wed, 18 Feb 2015 23:16:09 -0000",
        "Wed, 18 Feb 2015 23:16:09 +2359",
        "Wed, 18 Feb 2015 23:16:09 +2400",
        "Wed, 18 Feb 2015 23:16:09 -9959",
        "Wed, 18 Feb 2015 23:16:09 +0060",
        "Wed, 18 Feb 2015 23:16:09 +00:00",
        "Wed, 18 Feb 2015 23:16:09 +000",
        "Wed, 18 Feb 2015 23:16:09 \u{2212}0500",
        "Wed, 18 Feb 2015 23:16:09 +0000 (UTC)",
        "Wed, 18 Feb 2015 23:16:09 +0000 (a (nested) comment) (two)",
        "Wed, 18 Feb 2015 23:16:09 +0000 (unterminated",
        "Wed, 18 Feb 2015 23:16:09 +0000 (esc\\))",
        "Wed, 18 Feb 2015 23:16:09 +0000 ",
        "Wed, 18 Feb 2015 23:16:09 +0000x",
        "Wed,18 Feb 2015 23:16:09 GMT",
        "Wed , 18 Feb 2015 23:16:09 GMT",
        "  Wed, 18 Feb 2015 23:16:09 GMT",
        "18 Feb 2015 23:16:09 GMT",
        "Thu, 18 Feb 2015 23:16:09 GMT",
        "Wednesday, 18 Feb 2015 23:16:09 GMT",
        "Wed, 18 February 2015 23:16:09 GMT",
        "Wed, 18 fEB 2015 23:16:09 GMT",
        "Wed, 18 Feb 15 23:16:09 GMT",
        "Sat, 18 Feb 50 23:16:09 GMT",
        "Wed, 18 Feb 115 23:16:09 GMT",
        "Wed, 18 Feb 0015 23:16:09 GMT",
        "Wed, 18 Feb 2015 23:16 GMT",
        "Wed, 18 Feb 2015 23 : 16 : 09 GMT",
        "Wed, 18 Feb 2015 23:16: 09 GMT",
        "Wed, 18 Feb 2015 3:16:09 GMT",
        "Wed, 18 Feb 2015 24:16:09 GMT",
        "Wed, 18 Feb 2015 23:60:09 GMT",
        "Wed, 18 Feb 2015 23:59:60 GMT",
        "Wed, 18 Feb 2015 23:59:61 GMT",
        "Wed, 31 Feb 2015 23:16:09 GMT",
        "Wed, 32 Feb 2015 23:16:09 GMT",
        "Wed, 0 Feb 2015 23:16:09 GMT",
        "Wed, 18 Feb 2015 23:16:09",
        "Wed, 18 Feb 2015 23:16:09 ",
        "Fri, 31 Dec 9999 23:59:59 GMT",
        "Fri, 31 Dec 9999 23:59:59 -0100",
        "Sat, 01 Jan 10000 00:00:00 GMT",
        "Mon, 01 Jan 0001 00:00:00 +0100",
        "18\tFeb\t2015\t23:16:09\tGMT",
        "18\u{3000}Feb 2015 23:16:09 GMT",
        "Wed, 18 Feb 99999999999999999999 23:16:09 GMT",
        "",
        "W",
        "Wed",
        "Wed,",
        "Tue, 05 Mar 2024 12:00:00 GMT",
    ];

    #[test]
    fn rfc2822_matches_chrono() {
        for input in RFC2822_CASES {
            check(input, parse_rfc2822, chrono_rfc2822);
        }
        let ts = parse_rfc2822("Wed, 18 Feb 2015 23:16:09 GMT").unwrap();
        assert_eq!(ts.as_second(), 1424301369);
        // A leap second is clamped to :59
        let leap = parse_rfc2822("Wed, 18 Feb 2015 23:59:60 GMT").unwrap();
        assert_eq!(leap.to_string(), "2015-02-18T23:59:59Z");
    }

    const RFC3339_CASES: &[&str] = &[
        "2022-08-06T18:01:34Z",
        "2022-08-06T18:01:34z",
        "2022-08-06t18:01:34Z",
        "2022-08-06 18:01:34Z",
        "2022-08-06  18:01:34Z",
        "2022-08-06T18:01:34.000Z",
        "2022-08-06T18:01:34.123Z",
        "2022-08-06T18:01:34.123456789Z",
        "2022-08-06T18:01:34.1234567891234Z",
        "2022-08-06T18:01:34.Z",
        "2022-08-06T18:01:34.",
        "2022-08-06T18:01:34+00:00",
        "2022-08-06T18:01:34-07:30",
        "2022-08-06T18:01:34+0730",
        "2022-08-06T18:01:34+07 30",
        "2022-08-06T18:01:34+07:::30",
        "2022-08-06T18:01:34+07",
        "2022-08-06T18:01:34+07:3",
        "2022-08-06T18:01:34+07:60",
        "2022-08-06T18:01:34+23:59",
        "2022-08-06T18:01:34+24:00",
        "2022-08-06T18:01:34+99:00",
        "2022-08-06T18:01:34\u{2212}05:00",
        "2022-08-06T18:01:34 UTC",
        "2022-08-06T18:01:34utc",
        "2022-08-06T18:01:34 Z ",
        "2022-08-06T18:01:34 Z x",
        "2022-08-06T18:01:34",
        "2022-08-06T18:01:60Z",
        "2022-08-06T18:01:61Z",
        "2022-08-06T24:01:34Z",
        "2022-02-30T18:01:34Z",
        "2022-13-06T18:01:34Z",
        "2022-8-6T8:1:4Z",
        "2022 - 08 - 06T18 : 01 : 34Z",
        " 2022-08-06T18:01:34Z",
        "+2022-08-06T18:01:34Z",
        "-0001-08-06T18:01:34Z",
        "+10000-01-01T00:00:00Z",
        "+9999-12-31T23:59:59Z",
        "+9999-12-31T23:59:59-01:00",
        "12345-08-06T18:01:34Z",
        "2022-08-06X18:01:34Z",
        "2022-08-06",
        "2022-08-06T",
        "",
        "2009-02-13T23:31:30.0Z",
        "1970-01-01T00:00:00.000Z",
    ];

    #[test]
    fn rfc3339_matches_chrono() {
        for input in RFC3339_CASES {
            check(input, parse_rfc3339, chrono_rfc3339);
        }
        let ts = parse_rfc3339("2022-08-06T18:01:34.5-01:00").unwrap();
        assert_eq!(ts.to_string(), "2022-08-06T19:01:34.5Z");
    }

    const RFC1123_CASES: &[&str] = &[
        "Sun, 06 Nov 1994 08:49:37 GMT",
        "sun, 06 nov 1994 08:49:37 GMT",
        "Sun, 6 Nov 1994 8:49:37 GMT",
        "Sun,06 Nov 1994 08:49:37 GMT",
        "Sun,  06  Nov  1994  08:49:37  GMT",
        "Sun, 06 Nov 1994 08:49:37GMT",
        "Sun, 06 Nov 1994 08:49:37 gmt",
        "Sun, 06 Nov 1994 08:49:37 UTC",
        "Sun, 06 Nov 1994 08:49:37 GMT ",
        "Sun, 06 Nov 1994 08:49:37 GM",
        "Sun, 06 Nov 1994 08:49:37",
        "Sunday, 06 Nov 1994 08:49:37 GMT",
        "Sun, 06 November 1994 08:49:37 GMT",
        "Mon, 06 Nov 1994 08:49:37 GMT",
        "Sun, 06 Nov 94 08:49:37 GMT",
        "Sun, 06 Nov +1994 08:49:37 GMT",
        "Sun, 06 Nov 19940 08:49:37 GMT",
        "Sun, 06 Nov 1994 08:49 GMT",
        "Sun, 06 Nov 1994 08:49:60 GMT",
        "Sun, 06 Nov 1994 08: 49: 37 GMT",
        "Sun, 31 Nov 1994 08:49:37 GMT",
        "Fri, 31 Dec 9999 23:59:59 GMT",
        "06 Nov 1994 08:49:37 GMT",
        "",
        "Su",
    ];

    #[test]
    fn rfc1123_matches_chrono() {
        for input in RFC1123_CASES {
            check(input, parse_rfc1123, chrono_rfc1123);
        }
    }

    const AZ_CLI_CASES: &[&str] = &[
        "2023-01-25 12:34:56.123456",
        "2023-01-25 12:34:56.000000",
        "2023-1-5 2:4:6.123456",
        "2023-01-25  12:34:56.123456",
        "2023-01-25T12:34:56.123456",
        "2023-01-25 12:34:56.12345",
        "2023-01-25 12:34:56.1234567",
        "2023-01-25 12:34:56",
        "2023-01-25 12:34:56.",
        "2023-01-25 12:34:60.123456",
        "+2023-01-25 12:34:56.123456",
        "2023-02-29 12:34:56.123456",
        "",
    ];

    #[test]
    fn az_cli_matches_chrono() {
        for input in AZ_CLI_CASES {
            check(input, ours_az_cli, chrono_az_cli);
        }
    }

    /// Characters likely to exercise the grammars when spliced into valid inputs
    const ALPHABET: &[char] = &[
        '0', '1', '2', '5', '6', '9', ' ', '\t', ':', '-', '+', '.', ',', 'T', 'Z', 'z', 'G', 'M',
        'U', 'C', '(', ')', '\\', 'a', 'x', '\u{2212}', '\u{3000}', 'é',
    ];

    fn mutate(rng: &mut StdRng, input: &str) -> String {
        let mut chars: Vec<char> = input.chars().collect();
        for _ in 0..rng.random_range(1..=3) {
            let c = ALPHABET[rng.random_range(0..ALPHABET.len())];
            match rng.random_range(0..3) {
                0 if !chars.is_empty() => {
                    let i = rng.random_range(0..chars.len());
                    chars[i] = c;
                }
                1 if !chars.is_empty() => {
                    chars.remove(rng.random_range(0..chars.len()));
                }
                _ => chars.insert(rng.random_range(0..=chars.len()), c),
            }
        }
        chars.into_iter().collect()
    }

    /// Random timestamps in jiff's range, rendered in the given layouts, and mutated
    #[test]
    fn fuzz_against_chrono() {
        let mut rng = StdRng::seed_from_u64(0x0b1ec7);
        let suites: [(&[&str], Ours, Oracle); 4] = [
            (RFC2822_CASES, parse_rfc2822, chrono_rfc2822),
            (RFC3339_CASES, parse_rfc3339, chrono_rfc3339),
            (RFC1123_CASES, parse_rfc1123, chrono_rfc1123),
            (AZ_CLI_CASES, ours_az_cli, chrono_az_cli),
        ];
        let mut compared = 0usize;
        for _ in 0..4_000 {
            let secs = rng.random_range(-62_135_596_800i64..253_402_300_800);
            let nanos = rng.random_range(0..1_000_000_000u32);
            let dt = Utc.timestamp_opt(secs, nanos).unwrap();
            let generated = [
                dt.to_rfc2822(),
                dt.to_rfc3339_opts(SecondsFormat::AutoSi, rng.random()),
                dt.format("%a, %d %h %Y %T GMT").to_string(),
                dt.format("%Y-%m-%d %H:%M:%S.%6f").to_string(),
            ];
            for ((cases, ours, oracle), valid) in suites.iter().zip(&generated) {
                compared += usize::from(check(valid, *ours, *oracle));
                let seed = if rng.random_bool(0.5) {
                    valid.as_str()
                } else {
                    cases[rng.random_range(0..cases.len())]
                };
                for _ in 0..4 {
                    compared += usize::from(check(&mutate(&mut rng, seed), *ours, *oracle));
                }
            }
        }
        assert!(compared > 70_000, "only {compared} inputs compared");
    }

    #[test]
    fn formatting_matches_chrono() {
        let mut rng = StdRng::seed_from_u64(42);
        let fixed = [
            0i64,
            1_659_808_894,
            1_369_353_600,
            -1,
            -62_135_596_800,
            -62_135_596_801,
        ];
        // exclusive of jiff's extreme seconds, which only allow some nanoseconds
        let (min, max) = (Timestamp::MIN.as_second() + 1, Timestamp::MAX.as_second());
        let random = (0..2_000).map(|_| rng.random_range(min..max));
        for secs in fixed.into_iter().chain(random) {
            for nanos in [0u32, 500_000_000, 123_000, 1, 999_999_999] {
                let c = Utc.timestamp_opt(secs, nanos).unwrap();
                let j = Timestamp::new(secs, nanos as i32).unwrap();
                assert_eq!(
                    basic_datetime(j).to_string(),
                    c.format("%Y%m%dT%H%M%SZ").to_string()
                );
                assert_eq!(basic_date(j).to_string(), c.format("%Y%m%d").to_string());
                assert_eq!(
                    http_date(j).to_string(),
                    c.format("%a, %d %b %Y %H:%M:%S GMT").to_string()
                );
                assert_eq!(
                    http_date(j).to_string(),
                    c.format("%a, %d %h %Y %T GMT").to_string()
                );
                if c.format("%Y").to_string().len() == 4 {
                    // chrono's RFC 3339 writer only supports years 0..=9999
                    assert_eq!(
                        rfc3339_secs(j).to_string(),
                        c.to_rfc3339_opts(SecondsFormat::Secs, true)
                    );
                }
            }
        }
    }
}
