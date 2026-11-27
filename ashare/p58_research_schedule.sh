#!/usr/bin/env bash
# Run supplementary research only. The existing P55 launcher owns its own execution.
set -u
export TZ=Asia/Shanghai
umask 077
if [[ $# != 2 || ! -d $2 || ( $1 != close && $1 != post ) ]]; then
    echo "Usage: $0 {close|post} ABSOLUTE_RESEARCH_DIRECTORY" >&2
    exit 64
fi
phase=$1
root=$(cd -- "$2" && pwd -P) || exit 1
reports=$(cd -- "$root/../.." && pwd -P) || exit 1
previous=${ALPHAGPT_P5_NEXT_ROOT:-$reports/ashare_p5_next/pnext.bT5Erw}
follow=${ALPHAGPT_P5_FOLLOW_ROOT:-$reports/ashare_p5_follow/pfollow.yFSC8U}
legacy=$previous/ledger
panel=$follow/panel
today=$(date +%F)
hour=$(date +%H)
hhmm=$(date +%H%M)
if [[ $phase == post && $hour < 18 ]]; then
    echo "$(date -Is) BEFORE_POST_CLOSE_WINDOW"
    exit 0
fi

logs=$root/schedule-logs/$today
mkdir -p -- "$logs" || exit 1
exec 8>"$root/schedule-$phase.lock" || exit 1
if ! flock -n 8; then
    echo "$(date -Is) $phase ALREADY_RUNNING"
    exit 0
fi
run_log=$(mktemp -- "$logs/$phase.XXXXXX.log") || exit 1
echo "$(date -Is) $phase $run_log"
result=0

record() {
    printf '%s %s %s\n' "$(date -Is)" "$1" "$2" >> "$run_log"
}

step() {
    local name=$1 rc
    shift
    if [[ $(date +%F) != "$today" ]]; then
        record "$name" SESSION_DATE_CHANGED
        result=1
        return
    fi
    record "$name" START
    /bin/bash "$root/mode.sh" "$@" >> "$run_log" 2>&1
    rc=$?
    record "$name" "EXIT=$rc"
    if [[ $rc -ne 0 ]]; then result=1; fi
}

if [[ $phase == close ]]; then
    step fund_close ashare.p58_fund_snapshot --out "$root/fund" --phase close
else
    step fund_evening ashare.p58_fund_snapshot --out "$root/fund" --phase next-open
    step monitor ashare.p58_monitor --out "$root/monitor" --legacy "$legacy" --panel "$panel"
    p55_ready=false
    if [[ -f $legacy/timely/$today/manifest.json && -f $panel/timely/$today/manifest.json &&
          -f $panel/reports/$today/panel_report_audit.json ]]; then
        p55_ready=true
        if [[ -f $root/group_c/policy.json ]]; then
            group_args=(--out "$root/group_c" --panel "$panel" --legacy "$legacy")
            step group_c_capture ashare.p58_group_c capture "${group_args[@]}" \
                --model "$follow/ablation/models/group_c" --source "$previous/paired18"
        else
            record group_c_capture AWAITING_POLICY_FREEZE
        fi
    else
        record predictions AWAITING_TODAY_ARCHIVED_P55_MANIFESTS
    fi
    if [[ -f $root/group_c/policy.json && ( -f $root/group_c/timely/$today/manifest.json || $hhmm > 2329 ) ]]; then
        step group_c_report ashare.p58_group_c report --out "$root/group_c" --panel "$panel" \
            --legacy "$legacy" --asof "$today"
    else
        record group_c_report AWAITING_POLICY_OR_CAPTURE
    fi
    # Positions at today's open use prior-day archives. Allow the final nightly
    # attempt even if today's prediction or company-action observation is missing.
    if [[ -f $root/portfolio/policy.json && ( $p55_ready == true || $hhmm > 2329 ) ]]; then
        step portfolio ashare.p6_portfolio_run run --out "$root/portfolio" --legacy "$legacy" \
            --panel "$panel" --group "$root/group_c" --track forward --end "$today"
    else
        record portfolio AWAITING_POLICY_OR_FINAL_FORMAL_PROBE
    fi
fi
record "$phase" "EXIT=$result"
exit "$result"
