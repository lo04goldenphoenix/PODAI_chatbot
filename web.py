import streamlit as st
import pandas as pd
import json
import asyncio
from pymilvus import connections, Collection, utility
import warnings
import time
from datetime import datetime, timedelta

# Import modules
from ui.chatbot_interface import create_chatbot_interface
from data.data_processor import connect_to_milvus

# Import chatbot - đảm bảo file này có trong cùng thư mục
try:
    from chatbot import RnDChatbot

    CHATBOT_AVAILABLE = True
except ImportError:
    CHATBOT_AVAILABLE = False

warnings.filterwarnings('ignore')


# ==================== CACHING CONFIGURATION ====================

# Cache connection status để tránh kết nối lại liên tục
@st.cache_data(ttl=1800)  # Cache 30 phút
def get_connection_status():
    """Cache connection status để tránh kiểm tra liên tục"""
    try:
        return connect_to_milvus()
    except Exception as e:
        return False


# Cache chatbot initialization
@st.cache_resource
def initialize_cached_chatbot():
    """Initialize chatbot với resource caching - chỉ khởi tạo 1 lần"""
    if CHATBOT_AVAILABLE:
        try:
            return RnDChatbot()
        except Exception as e:
            st.error(f"❌ Lỗi khởi tạo chatbot: {e}")
            return None
    return None


# Cache CSS loading
@st.cache_data
def load_cached_css():
    """Cache CSS để tránh load lại mỗi lần"""
    from ui.styles import load_custom_css
    return load_custom_css()


# Cache static content
@st.cache_data
def get_header_content():
    """Cache nội dung header tĩnh"""
    return """
    <div class="main-header">
        <h1>🚀 Milvus Product Analytics Dashboard</h1>
        <p>🤖 RnD Assistant - Chatbot phân tích dữ liệu từ Milvus Vector Database</p>
    </div>
    """


# ==================== SESSION STATE MANAGEMENT ====================

def initialize_session_state():
    """Khởi tạo session state một cách tối ưu"""
    # Chỉ khởi tạo nếu chưa có
    if 'app_initialized' not in st.session_state:
        st.session_state.app_initialized = True
        st.session_state.last_connection_check = None
        st.session_state.connection_status = None

    if 'chatbot_initialized' not in st.session_state:
        st.session_state.chatbot_initialized = False
        st.session_state.chatbot = None
        st.session_state.chat_history = []
        st.session_state.chatbot_loading = False

    # Performance tracking
    if 'page_load_count' not in st.session_state:
        st.session_state.page_load_count = 0
        st.session_state.last_activity = datetime.now()

    st.session_state.page_load_count += 1
    st.session_state.last_activity = datetime.now()


# ==================== OPTIMIZED CHATBOT INITIALIZATION ====================

def setup_chatbot_optimized():
    """Thiết lập chatbot với caching tối ưu"""
    if not CHATBOT_AVAILABLE:
        if st.session_state.page_load_count == 1:  # Chỉ hiện warning lần đầu
            st.warning("⚠️ Không thể import chatbot. Chức năng chatbot sẽ bị vô hiệu hóa.")
        return False

    # Chỉ khởi tạo nếu chưa có và không đang loading
    if not st.session_state.chatbot_initialized and not st.session_state.chatbot_loading:
        st.session_state.chatbot_loading = True
    return st.session_state.chatbot_initialized


# ==================== PERFORMANCE METRICS ====================

@st.cache_data(ttl=60)  # Cache 1 phút
def get_performance_metrics():
    """Cache performance metrics"""
    return {
        'connection_checks': st.session_state.get('connection_checks', 0),
        'page_loads': st.session_state.page_load_count,
        'last_activity': st.session_state.last_activity.strftime("%H:%M:%S")
    }


# ==================== MAIN APPLICATION ====================

def main():
    """Main application với caching tối ưu"""

    # Cấu hình trang - chỉ chạy 1 lần
    st.set_page_config(
        page_title="Milvus Product Analytics Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Initialize session state
    initialize_session_state()

    # Load CSS (cached)
    load_cached_css()

    # Header (cached content)
    st.markdown(get_header_content(), unsafe_allow_html=True)



    # Setup chatbot
    setup_chatbot_optimized()

    # Main interface
    create_chatbot_interface()

    # Auto-cleanup old chat history (tùy chọn)
    cleanup_old_data()


# ==================== CLEANUP FUNCTIONS ====================

def cleanup_old_data():
    """Cleanup dữ liệu cũ để tránh memory leak"""
    # Giới hạn chat history
    max_history = 50
    if len(st.session_state.chat_history) > max_history:
        st.session_state.chat_history = st.session_state.chat_history[-max_history:]

    # Reset connection check nếu app không hoạt động lâu
    if st.session_state.last_activity < datetime.now() - timedelta(hours=2):
        st.session_state.last_connection_check = None
        st.session_state.connection_status = None


# ==================== ERROR HANDLING ====================

def handle_app_errors():
    """Xử lý lỗi app với graceful degradation"""
    try:
        main()
    except Exception as e:
        st.error(f"""
        ❌ **Đã xảy ra lỗi ứng dụng:**

        `{str(e)}`

        **Khắc phục:**
        1. Refresh trang (F5)
        2. Xóa cache trình duyệt
        3. Kiểm tra kết nối mạng
        """)

        # Reset session state nếu có lỗi nghiêm trọng
        if st.button("🔄 Reset ứng dụng"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()


if __name__ == "__main__":
    handle_app_errors()