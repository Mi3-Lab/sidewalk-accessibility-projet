#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${PORT:-8080}"
HOST="${HOST:-0.0.0.0}"
LOG_DIR="${ROOT}/logs"
PID_FILE="${LOG_DIR}/user_study_app_${PORT}.pid"
LOG_FILE="${LOG_DIR}/user_study_app_${PORT}.log"
HOST_FILE="${LOG_DIR}/user_study_app_${PORT}.host"
SESSION_NAME="accessibility_study_${PORT}"

mkdir -p "${LOG_DIR}"

print_access_help() {
  local host_name
  host_name="$(hostname -f 2>/dev/null || hostname)"

  echo
  echo "Best access from your laptop:"
  echo "  1. Keep this server process running."
  echo "  2. In a terminal on your laptop, run:"
  echo "     ssh -N -L ${PORT}:127.0.0.1:${PORT} ${USER}@${host_name}"
  echo "  3. Open this in your laptop browser:"
  echo "     http://127.0.0.1:${PORT}/study/"
  echo
  echo "If ${host_name} does not resolve from your laptop, replace it with the login/IP you normally use for SSH."
  echo "Direct URLs like http://<server>:${PORT}/study/ may hang if the university firewall blocks inbound ports."
}

is_running() {
  if command -v tmux >/dev/null 2>&1 && tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    return 0
  fi
  [[ -f "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null
}

case "${1:-start}" in
  start)
    if is_running; then
      echo "Study app already running on port ${PORT}."
      print_access_help
      exit 0
    fi
    cd "${ROOT}"

    STUDY_SERVER="${ROOT}/apps/accessibility-study/server.py"
    if command -v tmux >/dev/null 2>&1; then
      tmux new-session -d -s "${SESSION_NAME}" -c "${ROOT}" \
        "PORT=${PORT} HOST=${HOST} python \"${STUDY_SERVER}\" >\"${LOG_FILE}\" 2>&1"
      echo "tmux:${SESSION_NAME}" >"${PID_FILE}"
    else
      PORT="${PORT}" HOST="${HOST}" nohup python "${STUDY_SERVER}" >"${LOG_FILE}" 2>&1 &
      echo "$!" >"${PID_FILE}"
    fi
    hostname -f 2>/dev/null >"${HOST_FILE}" || hostname >"${HOST_FILE}"
    sleep 0.5
    if ! is_running; then
      echo "Study app failed to start. Last log lines:"
      tail -40 "${LOG_FILE}" 2>/dev/null || true
      rm -f "${PID_FILE}"
      exit 1
    fi
    echo "Study app started on port ${PORT}."
    if command -v tmux >/dev/null 2>&1; then
      echo "tmux session: ${SESSION_NAME}"
    else
      echo "PID: $(cat "${PID_FILE}")"
    fi
    echo "Server-local check URL: http://127.0.0.1:${PORT}/study/"
    echo "Server URL, only if port ${PORT} is reachable through VPN: http://$(hostname -f 2>/dev/null || hostname):${PORT}/study/"
    if command -v hostname >/dev/null 2>&1; then
      for ip in $(hostname -I 2>/dev/null || true); do
        echo "IP URL, only if reachable through VPN: http://${ip}:${PORT}/study/"
      done
    fi
    echo "Log: ${LOG_FILE}"
    print_access_help
    ;;
  stop)
    if command -v tmux >/dev/null 2>&1 && tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
      tmux kill-session -t "${SESSION_NAME}"
      rm -f "${PID_FILE}" "${HOST_FILE}"
      echo "Study app stopped."
    elif [[ -f "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
      kill "$(cat "${PID_FILE}")"
      rm -f "${PID_FILE}" "${HOST_FILE}"
      echo "Study app stopped."
    else
      echo "No running study app found for port ${PORT}."
    fi
    ;;
  status)
    if is_running; then
      echo "Study app running on port ${PORT}."
      if command -v tmux >/dev/null 2>&1 && tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
        echo "tmux session: ${SESSION_NAME}"
      elif [[ -f "${PID_FILE}" ]]; then
        echo "PID: $(cat "${PID_FILE}")"
      fi
      echo "Server-local check URL: http://127.0.0.1:${PORT}/study/"
      print_access_help
    else
      if [[ -f "${HOST_FILE}" ]]; then
        started_host="$(cat "${HOST_FILE}")"
        current_host="$(hostname -f 2>/dev/null || hostname)"
        if [[ "${started_host}" != "${current_host}" ]]; then
          echo "Study app was last started on ${started_host}, but this shell is on ${current_host}."
          echo "Check status from ${started_host}, or start a separate instance on this node with:"
          echo "  ./run_user_study_app.sh start"
          echo
          echo "If it is still running on ${started_host}, tunnel from your laptop with:"
          echo "  ssh -N -L ${PORT}:127.0.0.1:${PORT} ${USER}@${started_host}"
          exit 0
        fi
      fi
      echo "Study app is not running from ${PID_FILE}."
    fi
    ;;
  *)
    echo "Usage: PORT=8080 $0 {start|stop|status}"
    exit 2
    ;;
esac
