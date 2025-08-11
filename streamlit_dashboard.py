"""
streamlit_dashboard_new.py
통합 관리 메인 파일 - 각 모듈을 import하여 UI/UX 통합 관리
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import warnings
import logging

# 분리된 모듈들 import
from future_prediction import show_future_prediction
from kpi_comparison import show_past_comparison
from sales_comparison import show_sales_based_prediction

# 로깅 레벨 설정으로 경고 메시지 줄이기
logging.getLogger('streamlit').setLevel(logging.ERROR)

# 경고 메시지 필터링
warnings.filterwarnings('ignore')

# 환경 변수 설정으로 Streamlit 경고 줄이기
os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
os.environ['STREAMLIT_SERVER_RUN_ON_SAVE'] = 'false'

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 페이지 설정 - 더 안정적인 설정으로 변경
st.set_page_config(
    page_title="이퀄베리 수요 예측 대시보드",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 제목
st.title("📊 이퀄베리 수요 예측 대시보드")
st.markdown("---")

# 사이드바 - 설정
st.sidebar.header("⚙️ 설정")

# 예측 모드 선택
prediction_mode = st.sidebar.radio(
    "예측 모드",
    ["미래 예측", "과거 예측 vs 실제값 비교(KPI 기반)", "과거 예측 vs 실제 비교(판매데이터 기반)"],
    index=0
)

# 동적 월 목록 생성 함수
def generate_month_options():
    """현재 날짜를 기준으로 미래 3개월과 과거 6개월을 동적으로 생성"""
    from datetime import datetime, timedelta
    import calendar
    
    current_date = datetime.now()
    current_year = current_date.year
    current_month = current_date.month
    
    # 미래 3개월 생성 (영어 형식)
    future_months = []
    for i in range(1, 4):
        future_date = current_date + timedelta(days=32*i)  # 다음달로 이동
        future_date = future_date.replace(day=1)  # 1일로 설정
        year_str = str(future_date.year)[-2:]  # 년도 뒤 2자리
        month_str = calendar.month_abbr[future_date.month]  # 월 약어
        future_months.append(f"{year_str}-{month_str}")
    
    # 과거 6개월 생성 (한국어 형식)
    past_months = []
    for i in range(6, 0, -1):  # 6개월 전부터 1개월 전까지
        past_date = current_date - timedelta(days=32*i)  # 이전달로 이동
        past_date = past_date.replace(day=1)  # 1일로 설정
        past_months.append(f"{past_date.year}년 {past_date.month}월")
    
    return future_months, past_months

# 동적 월 목록 생성
future_months, past_months = generate_month_options()

# 모드에 따른 월 선택 옵션
if prediction_mode == "미래 예측":
    selected_month = st.sidebar.selectbox(
        "예측 대상 월",
        future_months,
        index=0
    )
elif prediction_mode == "과거 예측 vs 실제값 비교(KPI 기반)":
    selected_month = st.sidebar.selectbox(
        "비교 대상 월",
        past_months,
        index=0
    )
else:  # 과거 예측 vs 실제 비교(판매데이터 기반)
    selected_month = st.sidebar.selectbox(
        "분석 기준 월",
        past_months,
        index=0
    )

# 기본 경로 목록 (데이터 로딩 전에 사용)
default_routes = [
    "Amazon(USA)", "B2B(GLOBAL)", "Shopee(PH)", "Shopee(MY)", 
    "Shopee(SG)", "Shopee(VN)", "Shopee(TW)", "TikTokShop(USA)", "Shopee(TH)"
]

selected_routes = st.sidebar.multiselect(
    "예측 경로 선택",
    default_routes,
    default=["Amazon(USA)", "B2B(GLOBAL)"]
)

# 과거 비교를 위한 추가 설정
if prediction_mode == "과거 예측 vs 실제값 비교(KPI 기반)":
    st.sidebar.markdown("---")
    st.sidebar.markdown("**과거 예측 정확도 분석**")
    accuracy_threshold = st.sidebar.slider(
        "정확도 임계값 (%)",
        min_value=70,
        max_value=130,
        value=70,
        step=5
    )
else:
    # 다른 모드에서도 accuracy_threshold를 정의하여 오류 방지
    accuracy_threshold = 70

# 데이터 로드 함수
@st.cache_data
def load_data():
    # 현재 스크립트 파일의 디렉토리 경로
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # CSV 파일들의 절대 경로
    product_info_path = os.path.join(script_dir, 'product_info.csv')
    sales_history_path = os.path.join(script_dir, 'sales_history.csv')
    kpi_history_path = os.path.join(script_dir, 'kpi_history.csv')
    
    product_info = pd.read_csv(product_info_path, encoding='utf-8')
    sales_history = pd.read_csv(sales_history_path, encoding='utf-8')
    kpi_history = pd.read_csv(kpi_history_path, encoding='utf-8')
    
    # 디버깅: 데이터 로딩 확인
    print(f"product_info 컬럼: {list(product_info.columns)}")
    print(f"sales_history 컬럼: {list(sales_history.columns)}")
    print(f"kpi_history 컬럼: {list(kpi_history.columns)}")
    
    # 데이터 전처리
    product_info['판매가'] = product_info['판매가'].astype(str).str.replace(',', '').astype(float)
    kpi_history['KPI매출'] = kpi_history['KPI매출'].astype(str).str.replace(',', '').astype(float)
    
    return product_info, sales_history, kpi_history

# 기존 예측 함수 (호환성 유지)
def estimate_demand(kpi_df, product_df, adjustment_df):
    """기존 예측 함수 - 호환성 유지용"""
    # 각 경로별로 예측 수행
    results = []
    
    for _, kpi_row in kpi_df.iterrows():
        route = kpi_row['경로']
        target_revenue = kpi_row['KPI매출']
        
        # 해당 경로의 제품들
        route_products = product_df[product_df['경로'] == route]
        
        # 총 제품 수
        total_products = len(route_products)
        
        if total_products == 0:
            continue
        
        # 각 제품별 동일한 비중으로 매출 분배
        equal_revenue_per_product = target_revenue / total_products
        
        for _, product in route_products.iterrows():
            # 예측 수량 계산
            if product['판매가'] > 0:
                predicted_quantity = equal_revenue_per_product / product['판매가']
            else:
                predicted_quantity = 0
            
            # 조정 계수 적용 (adjustment_df에서 해당 제품의 조정계수 찾기)
            adjustment_factor = 1.0  # 기본값
            matching_adjustment = adjustment_df[
                (adjustment_df['경로'] == route) & 
                (adjustment_df['제품명'] == product['제품명'])
            ]
            
            if not matching_adjustment.empty:
                adjustment_factor = matching_adjustment.iloc[0]['조정계수']
            
            # 최종 예측 수량 (조정계수 적용)
            final_predicted_quantity = predicted_quantity * adjustment_factor
            
            # 결과 저장
            results.append({
                '월': kpi_row['월'],
                '경로': route,
                '제품코드': product['제품코드'],
                '제품명': product['제품명'],
                '판매가': product['판매가'],
                '예측수량': max(0, int(final_predicted_quantity)),
                '조정계수': adjustment_factor,
                '제품별_예상매출': equal_revenue_per_product,
                '보정수량': max(0, int(final_predicted_quantity)),
                '최종_예측수량': max(0, int(final_predicted_quantity))
            })
    
    return pd.DataFrame(results)

# 메인 앱
def main():
    # 데이터 로드
    product_info, sales_history, kpi_history = load_data()
    
    if prediction_mode == "미래 예측":
        show_future_prediction(product_info, sales_history, kpi_history, selected_month, selected_routes)
    elif prediction_mode == "과거 예측 vs 실제값 비교(KPI 기반)":
        show_past_comparison(product_info, sales_history, kpi_history, selected_month, selected_routes, accuracy_threshold)
    else:  # 과거 예측 vs 실제 비교(판매데이터 기반)
        show_sales_based_prediction(product_info, sales_history, kpi_history, selected_month, selected_routes)

if __name__ == "__main__":
    main() 
