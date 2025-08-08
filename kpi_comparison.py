"""
kpi_comparison.py
KPI 기반 과거 예측 vs 실제값 비교 기능을 담당하는 모듈
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# future_prediction 모듈의 함수들을 import
from future_prediction import (
    estimate_demand_improved,
    get_relative_past_months
)

def calculate_m1_sales_based_forecast(target_month, routes, product_info, sales_history):
    """
    M-1 시점에서 판매데이터 기반 다음 달 수요 예측 함수
    현재 월 데이터를 제외하고 과거 판매 데이터만으로 예측
    """
    # 월 형식 매핑
    korean_months = {
        '25-Aug': '2025년 8월',
        '25-Sep': '2025년 9월',
        '25-Oct': '2025년 10월',
        '2025년 5월': '2025년 5월',
        '2025년 6월': '2025년 6월',
        '2025년 7월': '2025년 7월'
    }
    
    target_month_korean = korean_months.get(target_month, target_month)
    print(f"calculate_m1_sales_based_forecast: 예측 목표 월 = {target_month_korean}")
    
    # 비교 대상월 대비 상대적으로 과거 3개월 계산
    past_months = get_relative_past_months(target_month_korean, 3)
    print(f"calculate_m1_sales_based_forecast: 사용할 과거 월들 = {past_months}")
    
    # 과거 판매 데이터 필터링 (목표월 제외)
    past_sales = sales_history[
        (sales_history['월'].isin(past_months)) & 
        (sales_history['월'] != target_month_korean) &  # 목표월 제외
        (sales_history['경로'].isin(routes))
    ]
    
    print(f"calculate_m1_sales_based_forecast: 과거 판매 데이터 행수 = {len(past_sales)}")
    print(f"calculate_m1_sales_based_forecast: 사용 가능한 월들 = {past_sales['월'].unique()}")
    
    # 과거 3개월 평균 판매량 계산
    result_data = []
    
    for route in routes:
        route_sales = past_sales[past_sales['경로'] == route]
        route_products = product_info[product_info['경로'] == route]
        
        print(f"calculate_m1_sales_based_forecast: 경로 {route} - 과거 판매 데이터 {len(route_sales)}개, 제품 {len(route_products)}개")
        
        for _, product in route_products.iterrows():
            product_code = product['제품코드']
            product_name = product['제품명']
            
            # 제품코드 기반 매칭 (먼저 시도)
            if '제품코드' in route_sales.columns and pd.notna(product_code):
                product_sales = route_sales[route_sales['제품코드'] == product_code]
            else:
                # 제품명 기반 매칭 (대안)
                product_sales = route_sales[route_sales['제품명'] == product_name]
            
            if len(product_sales) > 0:
                # 과거 3개월 평균 판매량 계산
                avg_quantity = product_sales['판매수량'].mean()
                # 최소 0으로 제한
                predicted_quantity = max(0, avg_quantity)
            else:
                # 판매 이력이 없는 제품은 0으로 예측
                predicted_quantity = 0
            
            result_data.append({
                '월': target_month,
                '경로': route,
                '제품코드': product_code,
                '제품명': product_name,
                'M1_예측수량': int(predicted_quantity)
            })
    
    result_df = pd.DataFrame(result_data)
    print(f"calculate_m1_sales_based_forecast: 결과 데이터 프레임 크기 = {len(result_df)}, 총 예측수량 = {result_df['M1_예측수량'].sum() if len(result_df) > 0 else 0}")
    return result_df

def compare_past_prediction(month, routes, product_info, sales_history, kpi_history):
    """과거 예측 vs 실제값 비교 함수"""
    # 월 형식 변환 (영어 ↔ 한국어)
    month_mapping = {
        '25-Aug': '2025년 8월',
        '25-Sep': '2025년 9월', 
        '25-Oct': '2025년 10월',
        '2025년 5월': '2025년 5월',
        '2025년 6월': '2025년 6월',
        '2025년 7월': '2025년 7월'
    }
    month_korean = month_mapping.get(month, month)
    
    # 해당 월의 KPI 데이터
    kpi_data = kpi_history[kpi_history['월'] == month]
    kpi_data = kpi_data[kpi_data['경로'].isin(routes)]
    
    # 한국어 형식으로 실제 판매 데이터 조회
    actual_sales = sales_history[sales_history['월'] == month_korean]
    actual_sales = actual_sales[actual_sales['경로'].isin(routes)]
    
    # 예측 실행 (개선된 로직 사용) - 비교 대상월과 목표 월을 동일하게 설정
    forecast_data = estimate_demand_improved(kpi_data, product_info, sales_history, month_korean, kpi_history)
    
    # M-1 시점에서의 판매데이터 기반 예측 계산
    print(f"compare_past_prediction: 입력된 month={month}, 변환된 month_korean={month_korean}")
    print(f"compare_past_prediction: selected_routes={routes}")
    m1_forecast_data = calculate_m1_sales_based_forecast(month, routes, product_info, sales_history)
    
    # 실제 판매 데이터와 병합 (제품코드 기반 - sales_history와 product_info 매칭)
    if '제품코드' in forecast_data.columns and '제품코드' in actual_sales.columns:
        comparison_df = pd.merge(
            forecast_data,
            actual_sales[['경로', '제품코드', '판매수량']],
            on=['경로', '제품코드'],
            how='left'
        )
    else:
        comparison_df = pd.merge(
            forecast_data,
            actual_sales[['경로', '제품명', '판매수량']],
            on=['경로', '제품명'],
            how='left'
        )
    
    # M-1 예측 데이터 병합
    if '제품코드' in forecast_data.columns and '제품코드' in m1_forecast_data.columns:
        comparison_df = pd.merge(
            comparison_df,
            m1_forecast_data[['경로', '제품코드', 'M1_예측수량']],
            on=['경로', '제품코드'],
            how='left'
        )
    else:
        comparison_df = pd.merge(
            comparison_df,
            m1_forecast_data[['경로', '제품명', 'M1_예측수량']],
            on=['경로', '제품명'],
            how='left'
        )
    
    # M1_예측수량이 null인 경우 0으로 채우기
    comparison_df['M1_예측수량'] = comparison_df['M1_예측수량'].fillna(0)
    
    # 실제 판매수량이 null인 경우 0으로 채우기
    comparison_df['판매수량'] = comparison_df['판매수량'].fillna(0)
    
    # 최종예측수량을 보정수량으로 변경 (정확한 KPI 기반 예측)
    comparison_df['보정수량'] = comparison_df['예측수량']
    
    # 정확도 계산
    comparison_df['예측_오차'] = abs(comparison_df['보정수량'] - comparison_df['판매수량'])
    
    # 예측 정확도 계산 (개선된 공식)
    comparison_df['예측_정확도'] = comparison_df.apply(
        lambda row: 100 - (abs(row['보정수량'] - row['판매수량']) / max(row['판매수량'], 1)) * 100 
        if row['판매수량'] > 0 
        else (100 if row['보정수량'] == 0 else 0), 
        axis=1
    )
    
    # 정확도를 0~100% 범위로 제한
    comparison_df['예측_정확도'] = comparison_df['예측_정확도'].clip(lower=0, upper=100)
    
    # 가중 정확도 계산 (수량 가중치 기반)
    total_actual = comparison_df['판매수량'].sum()
    if total_actual > 0:
        comparison_df['수량_가중치'] = comparison_df['판매수량'] / total_actual
        comparison_df['가중_정확도'] = comparison_df['예측_정확도'] * comparison_df['수량_가중치']
        # 가중 정확도도 0%에서 100% 사이로 제한
        comparison_df['가중_정확도'] = comparison_df['가중_정확도'].clip(lower=0, upper=100)
    else:
        comparison_df['수량_가중치'] = 0
        comparison_df['가중_정확도'] = 0
    
    return comparison_df

def show_past_comparison(product_info, sales_history, kpi_history, selected_month, selected_routes, accuracy_threshold=70):
    """KPI 기반 과거 예측 vs 실제값 비교 모드 메인 함수"""
    
    st.subheader(f"📊 {selected_month} 예측 vs 실제값 비교")
    
    # 디버깅 정보 표시 (월 형식이 통일되었으므로 매핑 불필요)
    st.info(f"🔍 비교 분석: {selected_month} 데이터 비교")
    
    # 과거 예측과 실제값 비교 (개선된 로직 사용)
    comparison_df = compare_past_prediction(selected_month, selected_routes, product_info, sales_history, kpi_history)
    
    # 선택된 과거 월의 KPI와 실제 수량 정보 표시
    st.subheader("📊 과거 월 KPI vs 실제 수량")
    
    # 월 형식 변환 (25-Jul -> 2025년 7월)
    month_mapping = {
        '25-May': '2025년 5월',
        '25-Jun': '2025년 6월', 
        '25-Jul': '2025년 7월'
    }
    past_month_formatted = month_mapping.get(selected_month, selected_month)
    
    # KPI 정보 가져오기
    kpi_data = kpi_history[
        (kpi_history['월'] == selected_month) & 
        (kpi_history['경로'].isin(selected_routes))
    ]
    
    # 실제 수량 정보 가져오기
    actual_sales_data = sales_history[
        (sales_history['월'] == past_month_formatted) & 
        (sales_history['경로'].isin(selected_routes))
    ]
    
    # 경로별 KPI vs 실제 수량 비교 테이블
    st.write("**📊 목표 KPI 수량 vs 실제 판매 수량 비교**")
    
    comparison_table_data = []
    
    if len(kpi_data) > 0 and len(actual_sales_data) > 0:
        for route in selected_routes:
            route_kpi = kpi_data[kpi_data['경로'] == route]
            route_actual = actual_sales_data[actual_sales_data['경로'] == route]
            
            kpi_quantity = 0
            actual_quantity = 0
            
            # KPI 수량 계산
            if len(route_kpi) > 0:
                kpi_value = pd.to_numeric(route_kpi['KPI매출'].iloc[0], errors='coerce')
                if pd.notna(kpi_value) and len(route_actual) > 0:
                    # 제품 정보와 결합하여 평균 판매가 계산
                    if '제품코드' in product_info.columns and '제품코드' in route_actual.columns:
                        route_with_price = route_actual.merge(
                            product_info[['경로', '제품코드', '판매가']], 
                            on=['경로', '제품코드'], 
                            how='left'
                        )
                    else:
                        route_with_price = route_actual.merge(
                            product_info[['경로', '제품명', '판매가']], 
                            on=['경로', '제품명'], 
                            how='left'
                        )
                    
                    avg_price = pd.to_numeric(route_with_price['판매가'].astype(str).str.replace(',', ''), errors='coerce').mean()
                    if avg_price > 0:
                        kpi_quantity = kpi_value / avg_price
            
            # 실제 수량 계산
            if len(route_actual) > 0:
                actual_quantity = pd.to_numeric(route_actual['판매수량'], errors='coerce').sum()
            
            # 달성률 계산
            achievement_rate = 0
            if kpi_quantity > 0:
                achievement_rate = (actual_quantity / kpi_quantity) * 100
            
            comparison_table_data.append({
                '경로': route,
                '목표 KPI 수량': f"{kpi_quantity:,.0f}개",
                '실제 판매 수량': f"{actual_quantity:,.0f}개",
                '달성률': f"{achievement_rate:.1f}%",
                '상태': '✅ 달성' if achievement_rate >= 100 else '❌ 미달성' if achievement_rate > 0 else '⚠️ 데이터 없음'
            })
        
        # 비교 테이블 표시
        comparison_df_display = pd.DataFrame(comparison_table_data)
        st.dataframe(comparison_df_display, use_container_width=True)
        
        # 요약 통계
        col1, col2, col3 = st.columns(3)
        with col1:
            total_kpi = sum([float(row['목표 KPI 수량'].replace('개', '').replace(',', '')) for row in comparison_table_data])
            st.metric("총 목표 KPI 수량", f"{total_kpi:,.0f}개")
        
        with col2:
            total_actual = sum([float(row['실제 판매 수량'].replace('개', '').replace(',', '')) for row in comparison_table_data])
            st.metric("총 실제 판매 수량", f"{total_actual:,.0f}개")
        
        with col3:
            overall_achievement = (total_actual / total_kpi * 100) if total_kpi > 0 else 0
            st.metric("전체 달성률", f"{overall_achievement:.1f}%")
    
    else:
        if len(kpi_data) == 0:
            st.warning(f"⚠️ {selected_month} KPI 데이터가 없습니다.")
        if len(actual_sales_data) == 0:
            st.warning(f"⚠️ {past_month_formatted} 실제 판매 데이터가 없습니다.")
    
    # KPI 달성률 계산 및 표시
    if len(kpi_data) > 0 and len(actual_sales_data) > 0:
        st.subheader("📈 KPI 달성률 분석 (수량 기준)")
        
        # KPI와 실제 수량 비교
        kpi_vs_actual = []
        for route in selected_routes:
            route_kpi = kpi_data[kpi_data['경로'] == route]
            route_actual = actual_sales_data[actual_sales_data['경로'] == route]
            
            if len(route_kpi) > 0 and len(route_actual) > 0:
                kpi_value = pd.to_numeric(route_kpi['KPI매출'].iloc[0], errors='coerce')
                actual_quantity = pd.to_numeric(route_actual['판매수량'], errors='coerce').sum()
                
                if pd.notna(kpi_value) and kpi_value > 0:
                    # 제품 정보와 결합하여 평균 판매가 계산
                    if '제품코드' in product_info.columns and '제품코드' in route_actual.columns:
                        route_with_price = route_actual.merge(
                            product_info[['경로', '제품코드', '판매가']], 
                            on=['경로', '제품코드'], 
                            how='left'
                        )
                    else:
                        route_with_price = route_actual.merge(
                            product_info[['경로', '제품명', '판매가']], 
                            on=['경로', '제품명'], 
                            how='left'
                        )
                    
                    avg_price = pd.to_numeric(route_with_price['판매가'].astype(str).str.replace(',', ''), errors='coerce').mean()
                    if avg_price > 0:
                        kpi_quantity = kpi_value / avg_price
                        achievement_rate = (actual_quantity / kpi_quantity) * 100
                        kpi_vs_actual.append({
                            '경로': route,
                            '목표_KPI_수량': kpi_quantity,
                            '실제_수량': actual_quantity,
                            '달성률': achievement_rate
                        })
        
        if kpi_vs_actual:
            achievement_df = pd.DataFrame(kpi_vs_actual)
            
            # 달성률 테이블
            achievement_display = achievement_df.copy()
            achievement_display['목표_KPI_수량'] = achievement_display['목표_KPI_수량'].apply(lambda x: f"{x:,.0f}개")
            achievement_display['실제_수량'] = achievement_display['실제_수량'].apply(lambda x: f"{x:,.0f}개")
            achievement_display['달성률'] = achievement_display['달성률'].round(1).astype(str) + '%'
            st.dataframe(achievement_display, use_container_width=True)
    
    # 정확도 요약 섹션 제거됨

    # 비교 차트
    st.subheader("📈 예측 vs 실제 수량 비교")
    fig = px.bar(
        comparison_df,
        x='제품명',
        y=['보정수량', '판매수량'],
        title="예측값 vs 실제값",
        barmode='group'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # 상세 비교 테이블
    st.subheader("📋 상세 비교 결과")
    
    # 보정수량 계산 로직 설명
    with st.expander("ℹ️ 보정수량 계산 로직 설명"):
        st.markdown("""
        **보정수량 계산 과정**:
        
        1. **순수 예측수량**: KPI매출 × 판매비중 ÷ 개별제품단가
           - 판매비중: 과거 3개월 실제 판매 데이터 기반 제품별 비중
           - 인기도 가중치 없이 순수한 예측량 계산
        
        2. **보정계수 계산 (3단계)**:
           - **1단계**: 과거 데이터 기반 기본 보정계수 계산
             - 과거 실제 판매량 ÷ 과거 순수 예측량
           - **2단계**: KPI 목표 맞추기 위한 스케일링 팩터 적용
             - 전체 KPI 목표 ÷ 전체 순수 예측 매출
           - **3단계**: 개별 제품 보정계수를 1.0 근처로 유지하면서 전체 목표 달성
        
        3. **보정수량**: 순수 예측수량 × 보정계수
        
        4. **최종 예측수량**: 보정수량 × 인기도 가중치 (최종 단계에서만 적용)
        
        **핵심 특징**:
        - ✅ **정확한 KPI 기반**: 제품별 예상매출의 총합이 경로 KPI와 정확히 일치
        - ✅ **순수한 보정계수**: 인기도 가중치 없이 실제 판매 패턴 기반 보정계수 계산
        - ✅ **과거 데이터 활용**: 실제 판매 이력을 바탕으로 한 정확한 예측
        - ✅ **동적 인기도 가중치**: 과거 판매 데이터 기반 동적 가중치 계산
        """)
    
    # 컬럼 순서 조정: 목표 KPI → KPI 기반 예측 → M-1 예측 → 실제 판매 순서
    comparison_display = comparison_df[['경로', '제품명', '예측수량', '보정수량', 'M1_예측수량', '판매수량', '예측_오차', '예측_정확도']].copy()
    
    # 숫자 컬럼에 콤마 적용
    comparison_display['예측수량'] = comparison_display['예측수량'].apply(lambda x: f"{x:,.0f}")
    comparison_display['보정수량'] = comparison_display['보정수량'].apply(lambda x: f"{x:,.0f}")
    comparison_display['M1_예측수량'] = comparison_display['M1_예측수량'].apply(lambda x: f"{x:,.0f}")
    comparison_display['판매수량'] = comparison_display['판매수량'].apply(lambda x: f"{x:,.0f}")
    comparison_display['예측_오차'] = comparison_display['예측_오차'].apply(lambda x: f"{x:,.1f}")
    comparison_display['예측_정확도'] = comparison_display['예측_정확도'].round(1).astype(str) + '%'
    
    # 컬럼명 변경
    comparison_display = comparison_display.rename(columns={
        '예측수량': '목표_KPI_수량',
        '보정수량': 'KPI_기반_예측',
        'M1_예측수량': 'M-1_판매데이터_예측',
        '판매수량': '실제_판매수량'
    })
    
    st.dataframe(comparison_display, use_container_width=True)
    
    # 정확도 분석 섹션 제거됨
