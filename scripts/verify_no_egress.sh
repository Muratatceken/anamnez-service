#!/usr/bin/env bash
# Kapalı devre kanıtı: N saniye boyunca LAN/yönetim/docker DIŞINA giden paket sayısını raporlar.
# Sunucuda root ile çalıştır. Beklenen: EGRESS-BLOCKED sayaçları artmıyor ve tcpdump 0 paket.
#   EDGE_IF=eth0 LAN=10.0.0.0/24 MGMT=10.0.99.0/24 scripts/verify_no_egress.sh 300
set -euo pipefail
DUR=${1:-60}
EDGE_IF=${EDGE_IF:-$(ip route | awk '/default/ {print $5; exit}')}
LAN=${LAN:-10.0.0.0/24}; MGMT=${MGMT:-10.0.99.0/24}
# Docker köprü alt ağları "dış" sayılmaz
FILTER="not net $LAN and not net $MGMT and not net 127.0.0.0/8 and not net 172.16.0.0/12 and not net 10.200.0.0/16"
echo "==> arayüz: ${EDGE_IF:-(bulunamadı)}  süre: ${DUR}s"
echo "==> nftables egress sayaçları (öncesi):"; nft list table inet closedloop 2>/dev/null | grep -E "EGRESS-BLOCKED" || true
echo "==> tcpdump (yalnızca uplink, LAN/yönetim/docker hariç):"
timeout "$DUR" tcpdump -ni "$EDGE_IF" "($FILTER)" -c 1000 2>/dev/null | tee /tmp/egress.log || true
echo "==> Yakalanan dış paket: $(wc -l < /tmp/egress.log | tr -d ' ')"
echo "==> nftables egress sayaçları (sonrası):"; nft list table inet closedloop 2>/dev/null | grep -E "EGRESS-BLOCKED" || true
