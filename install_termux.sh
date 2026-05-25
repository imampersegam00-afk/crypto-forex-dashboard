#!/bin/bash
# ============================================================
#  ELITE TERMINAL — Auto Install Script
#  Untuk Termux + Debian (proot) di Android
# ============================================================

echo ""
echo "======================================"
echo "  ELITE TERMINAL — INSTALL SCRIPT"
echo "======================================"
echo ""

# Update Debian packages
echo "[1/6] Update packages..."
apt update -y && apt upgrade -y

echo "[2/6] Install Python & tools..."
apt install -y python3 python3-pip git curl wget

echo "[3/6] Clone dari GitHub..."
cd ~
rm -rf crypto-forex-dashboard
git clone https://github.com/imampersegam00-afk/crypto-forex-dashboard.git
cd crypto-forex-dashboard

echo "[4/6] Install Python dependencies..."
pip3 install flask yfinance ta pandas numpy requests feedparser scikit-learn

echo "[5/6] Buat launcher..."
cat > ~/start_dashboard.sh << 'EOF'
#!/bin/bash
cd ~/crypto-forex-dashboard
echo ""
echo "======================================"
echo "  ELITE TERMINAL — STARTING..."
echo "======================================"
echo ""
echo "  Buka browser: http://localhost:8080"
echo "  Stop: tekan CTRL+C"
echo ""
python3 app.py
EOF
chmod +x ~/start_dashboard.sh

echo "[6/6] Selesai!"
echo ""
echo "======================================"
echo "  CARA JALANKAN:"
echo ""
echo "  bash ~/start_dashboard.sh"
echo ""
echo "  Lalu buka browser:"
echo "  http://localhost:8080"
echo "======================================"
echo ""
