#!/bin/bash

# ==========================================
# tc 网络限速 + 时延模拟脚本
# ==========================================
#
# 用法:
#   sudo ./net_limit.sh start
#   sudo ./net_limit.sh stop
#   sudo ./net_limit.sh status
#
# ==========================================

# 网卡名称
DEV="ens1f0"

# 带宽限制
RATE="5gbit"

# 突发缓冲
BURST="32kbit"

# 队列最大时延
LATENCY="400ms"

# 网络传播时延
DELAY="20ms"

start_tc() {
    echo "[INFO] 开始配置 tc ..."

    # 清理旧规则
    tc qdisc del dev ${DEV} root 2>/dev/null

    # 建立 HTB 根队列
    tc qdisc add dev ${DEV} root handle 1: htb default 10

    # 创建带宽限制类
    tc class add dev ${DEV} parent 1: classid 1:10 \
        htb rate ${RATE} ceil ${RATE}

    # 在该类下增加 netem 时延模拟
    tc qdisc add dev ${DEV} parent 1:10 handle 10: \
        netem delay ${DELAY}

    echo "[INFO] 配置完成"
}

stop_tc() {
    echo "[INFO] 删除 tc 配置 ..."
    tc qdisc del dev ${DEV} root 2>/dev/null
    echo "[INFO] tc 已恢复默认"
}

status_tc() {
    echo "========== qdisc =========="
    tc qdisc show dev ${DEV}

    echo
    echo "========== class =========="
    tc class show dev ${DEV}
}

case "$1" in
    start)
        start_tc
        ;;
    stop)
        stop_tc
        ;;
    status)
        status_tc
        ;;
    *)
        echo "Usage:"
        echo "  sudo $0 start"
        echo "  sudo $0 stop"
        echo "  sudo $0 status"
        exit 1
        ;;
esac