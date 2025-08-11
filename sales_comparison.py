"""
sales_comparison.py
판매데이터 기반 과거 예측 vs 실제 비교 기능을 담당하는 모듈
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def get_dynamic_past_months(analysis_period, current_month):
    """
    분석 기간에 따라 동적으로 과거 월을 설정합니다.
    """
    # 월 매핑 (미래 예측용)
    future_month_mapping = {
        '25-Aug': '2025년 8월',
        '25-Sep': '2025년 9월', 
        '25-Oct': '2025년 10월'
    }
    
    # 월 매핑 (과거 분석용)
    past_month_mapping = {
        '2025년 7월': '2025년 7월',
        '2025년 6월': '2025년 6월',
        '2025년 5월': '2025년 5월'
    }
    
    # 현재 월을 한국어로 변환
    if current_month in future_month_mapping:
        current_month_korean = future_month_mapping[current_month]
    elif current_month in past_month_mapping:
        current_month_korean = past_month_mapping[current_month]
    else:
        current_month_korean = current_month  # 이미 한국어인 경우
    
    # 모든 가능한 월 목록 (과거 데이터)
    all_months = ['2025년 2월', '2025년 3월', '2025년 4월', '2025년 5월', '2025년 6월', '2025년 7월']
    
    # 현재 월의 인덱스 찾기
    try:
        current_index = all_months.index(current_month_korean)
    except ValueError:
        # 현재 월이 과거 데이터에 없는 경우 (미래 예측)
        if current_month_korean in ['2025년 8월', '2025년 9월', '2025년 10월']:
            # 미래 예측의 경우 가장 최근 월(7월)을 기준으로 설정
            current_index = len(all_months) - 1
        else:
            current_index = len(all_months) - 1  # 기본값
    
    # 분석 기간에 따라 과거 월 설정
    if analysis_period == "3개월":
        start_index = max(0, current_index - 2)  # 현재 월 포함 3개월
    elif analysis_period == "6개월":
        start_index = max(0, current_index - 5)  # 현재 월 포함 6개월
    else:  # 12개월
        start_index = 0
    
    past_months = all_months[start_index:current_index + 1]  # 현재 월 포함
    
    # 최소 3개월은 보장
    if len(past_months) < 3:
        past_months = all_months[-3:]
    
    return past_months

def calculate_monthly_weights(past_months, weighting_method):
    """
    월별 가중치를 계산합니다.
    """
    n_months = len(past_months)
    
    if weighting_method == "최근 가중":
        # 최근 월일수록 높은 가중치 (지수 감소)
        weights = [0.5 ** (n_months - i - 1) for i in range(n_months)]
        # 정규화
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
    elif weighting_method == "계절성 가중":
        # 계절성 가중치를 균등 가중치로 대체
        weights = [1.0 / n_months] * n_months
        
    else:  # 균등 가중
        weights = [1.0 / n_months] * n_months
    
    weight_dict = dict(zip(past_months, weights))
    
    # 디버깅: 가중치 계산 결과 출력
    print(f"가중치 계산 - 방식: {weighting_method}, 월 수: {n_months}")
    for month, weight in weight_dict.items():
        print(f"  {month}: {weight:.3f}")
    
    return weight_dict

def apply_dynamic_change_rate_correction(change_rate, recent_sales, previous_sales, correction_strength):
    """
    보정 강도에 따른 동적 변화율 보정을 적용합니다.
    """
    # 보정 강도별 계수
    correction_factors = {
        "약함": {"high": 0.8, "medium": 0.9, "low": 1.0},
        "보통": {"high": 0.5, "medium": 0.7, "low": 0.9},
        "강함": {"high": 0.3, "medium": 0.5, "low": 0.7}
    }
    
    factors = correction_factors.get(correction_strength, correction_factors["보통"])
    
    # 원본 변화율의 부호 보존
    original_sign = 1 if change_rate >= 0 else -1
    abs_change_rate = abs(change_rate)
    
    # 평균 판매량 계산
    avg_sales = (recent_sales + previous_sales) / 2
    
    # 작은 스케일 기준
    small_scale_threshold = 1500
    
    # 보정 계수 계산
    if abs_change_rate > 50:  # 50% 이상의 큰 변화율
        if avg_sales <= small_scale_threshold:
            correction_factor = factors["high"]
        else:
            correction_factor = factors["medium"]
    elif abs_change_rate > 20:  # 20-50% 변화율
        if avg_sales <= small_scale_threshold:
            correction_factor = factors["medium"]
        else:
            correction_factor = factors["low"]
    else:
        correction_factor = 1.0  # 보정 없음
    
    corrected_abs_rate = abs_change_rate * correction_factor
    corrected_change_rate = original_sign * corrected_abs_rate
    
    return corrected_change_rate

def analyze_sales_trend_dynamic(pivot_data, past_months, monthly_weights, correction_strength):
    """
    동적 파라미터를 적용한 판매 데이터의 추세를 분석합니다.
    """
    trend_analysis = {}
    
    for product in pivot_data.index:
        sales_data = pivot_data.loc[product]
        
        # 가중 평균 계산
        weighted_recent_sales = 0
        weighted_previous_sales = 0
        recent_weight_sum = 0
        previous_weight_sum = 0
        
        # 선택된 월을 기준으로 최근과 이전으로 분할
        if len(past_months) >= 6:
            # 6개월 이상인 경우: 최근 3개월 vs 이전 3개월
            recent_months = past_months[-3:]  # 최근 3개월
            previous_months = past_months[-6:-3]  # 이전 3개월
        elif len(past_months) >= 4:
            # 4-5개월인 경우: 최근 2개월 vs 이전 2개월
            recent_months = past_months[-2:]  # 최근 2개월
            previous_months = past_months[-4:-2]  # 이전 2개월
        else:
            # 3개월인 경우: 최근 1개월 vs 이전 2개월
            recent_months = past_months[-1:]  # 최근 1개월
            previous_months = past_months[:-1]  # 이전 2개월
        
        # 가중 평균 계산
        for month in recent_months:
            weight = monthly_weights.get(month, 1.0)
            weighted_recent_sales += sales_data.get(month, 0) * weight
            recent_weight_sum += weight
        
        for month in previous_months:
            weight = monthly_weights.get(month, 1.0)
            weighted_previous_sales += sales_data.get(month, 0) * weight
            previous_weight_sum += weight
        
        # 정규화
        if recent_weight_sum > 0:
            weighted_recent_sales /= recent_weight_sum
        if previous_weight_sum > 0:
            weighted_previous_sales /= previous_weight_sum
        
        # 변화율 계산
        if weighted_previous_sales > 0:
            change_rate = ((weighted_recent_sales - weighted_previous_sales) / weighted_previous_sales) * 100
        else:
            change_rate = 0
        
        # 보정 강도에 따른 변화율 보정
        corrected_change_rate = apply_dynamic_change_rate_correction(
            change_rate, weighted_recent_sales, weighted_previous_sales, correction_strength
        )
        
        # 추세 판단 (보정된 변화율 사용)
        if corrected_change_rate > 5:
            trend = '상승'
        elif corrected_change_rate < -5:
            trend = '하락'
        else:
            trend = '안정'
        
        trend_analysis[product] = {
            'trend': trend,
            'change_rate': corrected_change_rate,
            'original_change_rate': change_rate,
            'weighted_recent_sales': weighted_recent_sales,
            'weighted_previous_sales': weighted_previous_sales,
            'monthly_weights': monthly_weights,
            'recent_months': recent_months,
            'previous_months': previous_months
        }
    
    return trend_analysis



def predict_future_sales_dynamic(trend_analysis, pivot_data, months_ahead, monthly_weights):
    """
    동적 파라미터를 적용한 향후 판매 예측을 수행합니다.
    """
    future_forecast = {}
    
    for product in pivot_data.index:
        trend_info = trend_analysis[product]
        change_rate = trend_info['change_rate']
        
        # 가중 평균 현재 판매량 계산
        current_sales = 0
        total_weight = 0
        
        for month in monthly_weights.keys():
            if month in pivot_data.columns:
                weight = monthly_weights[month]
                current_sales += pivot_data.loc[product, month] * weight
                total_weight += weight
        
        if total_weight > 0:
            current_sales /= total_weight
        
        # 월별 예측 계산 (추세에 따른 누적 변화율 적용)
        monthly_forecasts = []
        for month_idx in range(months_ahead):
            # 월별 누적 변화율 계산 (시간이 지날수록 변화가 누적됨)
            if change_rate > 0:  # 성장 추세
                # 성장 추세: 시간이 지날수록 더 성장 (가속화)
                cumulative_growth_rate = change_rate * (1 + month_idx * 0.1)  # 매월 10%씩 가속화
            elif change_rate < 0:  # 하향 추세
                # 하향 추세: 시간이 지날수록 더 하향 (가속화)
                cumulative_growth_rate = change_rate * (1 + month_idx * 0.1)  # 매월 10%씩 가속화
            else:  # 안정 추세
                cumulative_growth_rate = change_rate
            
            # 변화율 적용
            growth_factor = 1 + (cumulative_growth_rate / 100)
            
            # 예측 수량 계산
            predicted_sales = current_sales * growth_factor
            monthly_forecasts.append(max(0, predicted_sales))
        
        future_forecast[product] = {
            'trend': trend_info['trend'],
            'change_rate': change_rate,
            'total_forecast': sum(monthly_forecasts),
            'monthly_forecasts': monthly_forecasts
        }
    
    return future_forecast

def calculate_total_forecast_summary_dynamic(filtered_sales, selected_routes, past_months, monthly_weights, correction_strength):
    """
    동적 파라미터를 적용한 전체 예측 요약을 계산합니다.
    """
    summary = {}
    
    for route in selected_routes:
        route_sales = filtered_sales[filtered_sales['경로'] == route]
        
        if len(route_sales) == 0:
            continue
        
        # 제품별 월별 판매량 계산
        product_monthly_sales = route_sales.groupby(['제품명', '월'])['판매수량'].sum().reset_index()
        pivot_data = product_monthly_sales.pivot(index='제품명', columns='월', values='판매수량').fillna(0)
        
        # 동적 추세 분석 및 예측
        trend_analysis = analyze_sales_trend_dynamic(pivot_data, past_months, monthly_weights, correction_strength)
        future_forecast = predict_future_sales_dynamic(trend_analysis, pivot_data, 6, monthly_weights)
        
        # 월 평균 판매량 계산 (가중 평균 적용)
        products_info = {}
        
        for product in pivot_data.index:
            # 가중 평균 판매량 계산
            weighted_sales_sum = 0
            total_weight = 0
            
            for month in past_months:
                if month in pivot_data.columns:
                    weight = monthly_weights.get(month, 1.0)
                    weighted_sales_sum += pivot_data.loc[product, month] * weight
                    total_weight += weight
            
            monthly_avg_sales = weighted_sales_sum / total_weight if total_weight > 0 else 0
            
            # 6개월 예측의 월 평균 수량 계산
            monthly_avg_forecast = future_forecast[product]['total_forecast'] / 6
            
            products_info[product] = {
                'current_sales': monthly_avg_sales,  # 가중 평균 판매량
                'total_forecast': monthly_avg_forecast,  # 6개월 예측의 월 평균 수량
                'trend': future_forecast[product]['trend'],
                'change_rate': future_forecast[product]['change_rate'],
                'original_change_rate': future_forecast[product].get('original_change_rate', future_forecast[product]['change_rate']),
                'monthly_forecasts': future_forecast[product]['monthly_forecasts'],
                'weighted_analysis': True
            }
        
        summary[route] = products_info
    
    return summary

def filter_and_sort_forecast_results(total_forecast_summary):
    """0개 판매/예측 제품 제외 및 추세별 정렬"""
    filtered_summary = {}
    
    for route, products in total_forecast_summary.items():
        filtered_products = {}
        for product, info in products.items():
            # 현재 판매량이 0이거나 예측 판매량이 0인 제품 제외
            if info['current_sales'] > 0 and info['total_forecast'] > 0:
                filtered_products[product] = info
        
        if filtered_products:
            # 추세별 정렬 (상승, 안정, 하락 순)
            trend_order = {'상승': 0, '안정': 1, '하락': 2}
            sorted_products = dict(sorted(
                filtered_products.items(),
                key=lambda x: (trend_order[x[1]['trend']], -x[1]['change_rate'])
            ))
            filtered_summary[route] = sorted_products
    
    return filtered_summary

def display_product_trend_table(filtered_summary, analysis_month=None):
    """제품별 판매추세 및 예측 테이블 표시 (동적 분석 결과 포함)"""
    if not filtered_summary:
        st.warning("표시할 데이터가 없습니다.")
        return
    
    # 모든 제품 데이터를 하나의 테이블로 통합
    table_data = []
    high_change_rate_products = []  # 200% 이상 변화율 제품 추적
    
    for route, products in filtered_summary.items():
        for product, info in products.items():
            trend_icon = "📈" if info['trend'] == '상승' else "📉" if info['trend'] == '하락' else "➡️"
            
            # 원본 변화율과 보정된 변화율 표시
            original_change_rate = info.get('original_change_rate', info['change_rate'])
            corrected_change_rate = info['change_rate']
            
            # 200% 이상 변화율 체크
            is_high_change_rate = abs(corrected_change_rate) >= 200
            
            # 보정이 적용된 경우 표시 (소수 1자리까지)
            if abs(original_change_rate - corrected_change_rate) > 0.1:
                change_rate_display = f"{round(corrected_change_rate, 1)}% (보정됨)"
            else:
                change_rate_display = f"{round(corrected_change_rate, 1)}%"
            
            # 200% 이상인 경우 빨간색으로 표시
            if is_high_change_rate:
                change_rate_display = f"🔴 **{change_rate_display}**"
                high_change_rate_products.append(f"{route} - {product}")
            
            # 동적 분석 여부 확인
            is_weighted = info.get('weighted_analysis', False)
            analysis_type = "동적" if is_weighted else "기본"
            
            table_data.append({
                '경로': route,
                '제품명': product,
                '분석방식': analysis_type,
                '기준월': analysis_month if analysis_month else "N/A",
                '월 평균 판매량': f"{int(info['current_sales']):,}개",
                '추세': f"{trend_icon} {info['trend']}",
                '변화율': change_rate_display,
                '6개월 예측(월평균)': f"{int(info['total_forecast']):,}개"
            })
    
    # 200% 이상 변화율 제품이 있는 경우 경고 표시
    if high_change_rate_products:
        st.warning("⚠️ **정합성 유의**: 다음 제품들의 변화율이 200% 이상으로 급격한 변화를 보입니다. 데이터 정합성을 확인해주세요.")
        for product in high_change_rate_products:
            st.write(f"• {product}")
        st.markdown("---")
    
    df = pd.DataFrame(table_data)
    st.dataframe(df, use_container_width=True)
    
    # 동적 분석 통계
    dynamic_count = sum(1 for item in table_data if item['분석방식'] == '동적')
    basic_count = sum(1 for item in table_data if item['분석방식'] == '기본')
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("동적 분석 제품", f"{dynamic_count}개")
    with col2:
        st.metric("기본 분석 제품", f"{basic_count}개")

def display_monthly_forecast_chart(filtered_summary, filtered_sales, past_months):
    """판매 추이 및 월별 예측 수량 추이 그래프 표시 (경로/제품 선택 가능)"""
    if not filtered_summary:
        st.warning("표시할 데이터가 없습니다.")
        return
    
    # 보기 방식 선택
    view_type = st.radio(
        "보기 방식 선택:",
        ["경로별 전체", "제품별 개별", "제품별 경로 합계"],
        horizontal=True,
        help="경로별 전체: 각 경로의 모든 제품 합계, 제품별 개별: 특정 제품의 개별 추이, 제품별 경로 합계: 한 제품의 모든 경로 합계"
    )
    
    if view_type == "경로별 전체":
        # 경로별 전체 보기
        route_options = list(filtered_summary.keys())
        selected_route = st.selectbox(
            "보고 싶은 경로를 선택하세요:",
            ["전체 경로"] + route_options,
            index=0
        )
        
        if selected_route == "전체 경로":
            selected_routes = route_options
        else:
            selected_routes = [selected_route]
            
        display_route_summary_chart(filtered_summary, filtered_sales, past_months, selected_routes)
        
    elif view_type == "제품별 개별":
        # 제품별 개별 보기
        product_options = []
        for route, products in filtered_summary.items():
            for product in products.keys():
                product_options.append(f"{route} - {product}")
        
        selected_product = st.selectbox(
            "보고 싶은 제품을 선택하세요:",
            product_options,
            index=0
        )
        
        selected_route, selected_product_name = selected_product.split(" - ", 1)
        display_individual_product_chart(filtered_summary, filtered_sales, past_months, selected_route, selected_product_name)
        
    else:  # 제품별 경로 합계
        # 제품별 경로 합계 보기
        all_products = set()
        for route, products in filtered_summary.items():
            for product in products.keys():
                all_products.add(product)
        
        selected_product = st.selectbox(
            "보고 싶은 제품을 선택하세요 (모든 경로 합계):",
            sorted(list(all_products)),
            index=0
        )
        
        display_product_route_summary_chart(filtered_summary, filtered_sales, past_months, selected_product)

def display_route_summary_chart(filtered_summary, filtered_sales, past_months, selected_routes):
    """경로별 전체 제품 합계 차트 표시"""
    months = ['2025년 8월', '2025년 9월', '2025년 10월', '2025년 11월', '2025년 12월', '2026년 1월']
    
    fig = go.Figure()
    
    for route in selected_routes:
        if route not in filtered_summary:
            continue
            
        # 과거 판매 데이터 수집
        route_sales = filtered_sales[filtered_sales['경로'] == route]
        past_monthly_data = {}
        
        # 과거 6개월 판매 데이터
        for month in past_months:
            month_sales = route_sales[route_sales['월'] == month]['판매수량'].sum()
            past_monthly_data[month] = month_sales
        
        # 예측 데이터 수집
        route_monthly_data = {}
        for product, info in filtered_summary[route].items():
            monthly_forecasts = info['monthly_forecasts']
            for i, month in enumerate(months):
                if month not in route_monthly_data:
                    route_monthly_data[month] = 0
                route_monthly_data[month] += int(monthly_forecasts[i]) if i < len(monthly_forecasts) else 0
        
        # 과거 데이터 (실선)
        fig.add_trace(go.Scatter(
            x=list(past_monthly_data.keys()),
            y=list(past_monthly_data.values()),
            mode='lines+markers',
            name=f'{route} (과거)',
            line=dict(width=3),
            marker=dict(size=8)
        ))
        
        # 예측 데이터 (점선)
        fig.add_trace(go.Scatter(
            x=list(route_monthly_data.keys()),
            y=list(route_monthly_data.values()),
            mode='lines+markers',
            name=f'{route} (예측)',
            line=dict(width=3, dash='dash'),
            marker=dict(size=8, symbol='diamond')
        ))
    
    fig.update_layout(
        title=f'경로별 판매 추이 및 향후 6개월 예측',
        xaxis_title='월',
        yaxis_title='판매/예측 수량 (개)',
        hovermode='x unified',
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 경로별 월별 예측 수량 요약 테이블
    st.markdown("**경로별 월별 예측 수량 요약:**")
    summary_data = []
    for route in selected_routes:
        if route not in filtered_summary:
            continue
        route_monthly_data = {}
        for product, info in filtered_summary[route].items():
            monthly_forecasts = info['monthly_forecasts']
            for i, month in enumerate(months):
                if month not in route_monthly_data:
                    route_monthly_data[month] = 0
                route_monthly_data[month] += int(monthly_forecasts[i]) if i < len(monthly_forecasts) else 0
        
        for month, quantity in route_monthly_data.items():
            summary_data.append({
                '경로': route,
                '월': month,
                '예측 수량': f"{int(quantity):,}개"
            })
    
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True)

def display_individual_product_chart(filtered_summary, filtered_sales, past_months, selected_route, selected_product):
    """개별 제품 차트 표시"""
    months = ['2025년 8월', '2025년 9월', '2025년 10월', '2025년 11월', '2025년 12월', '2026년 1월']
    
    if selected_route not in filtered_summary or selected_product not in filtered_summary[selected_route]:
        st.warning("선택된 제품에 대한 데이터가 없습니다.")
        return
    
    info = filtered_summary[selected_route][selected_product]
    monthly_forecasts = info['monthly_forecasts']
    
    # 과거 판매 데이터 수집
    product_sales = filtered_sales[
        (filtered_sales['경로'] == selected_route) & 
        (filtered_sales['제품명'] == selected_product)
    ]
    past_monthly_data = {}
    
    # 과거 6개월 판매 데이터
    for month in past_months:
        month_sales = product_sales[product_sales['월'] == month]['판매수량'].sum()
        past_monthly_data[month] = month_sales
    
    # 예측 데이터
    forecast_monthly_data = {}
    for i, month in enumerate(months):
        forecast_monthly_data[month] = int(monthly_forecasts[i]) if i < len(monthly_forecasts) else 0
    
    # 그래프 생성
    fig = go.Figure()
    
    # 추세에 따른 색상 설정
    trend = info.get('trend', '안정')
    if trend == '상승':
        line_color = '#2E8B57'  # 녹색
    elif trend == '하락':
        line_color = '#DC143C'  # 빨간색
    else:
        line_color = '#1f77b4'  # 파란색
    
    # 과거 데이터 (실선)
    fig.add_trace(go.Scatter(
        x=list(past_monthly_data.keys()),
        y=list(past_monthly_data.values()),
        mode='lines+markers',
        name=f'{selected_product} (과거)',
        line=dict(color=line_color, width=3),
        marker=dict(size=8)
    ))
    
    # 예측 데이터 (점선)
    fig.add_trace(go.Scatter(
        x=list(forecast_monthly_data.keys()),
        y=list(forecast_monthly_data.values()),
        mode='lines+markers',
        name=f'{selected_product} (예측)',
        line=dict(color=line_color, width=3, dash='dash'),
        marker=dict(size=8, symbol='diamond')
    ))
    
    fig.update_layout(
        title=f'제품별 판매 추이 및 향후 6개월 예측 ({selected_route} - {selected_product})',
        xaxis_title='월',
        yaxis_title='판매/예측 수량 (개)',
        hovermode='x unified',
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 제품별 상세 정보 표시
    st.markdown(f"**제품 상세 정보:**")
    st.write(f"- **경로**: {selected_route}")
    st.write(f"- **제품명**: {selected_product}")
    st.write(f"- **추세**: {trend}")
    st.write(f"- **변화율**: {info.get('change_rate', 'N/A')}%")
    st.write(f"- **월 평균 판매량**: {int(info.get('current_sales', 0)):,}개")
    st.write(f"- **6개월 예측(월평균)**: {int(info.get('total_forecast', 0)):,}개")
    
    # 과거 판매량과 예측 수량 요약 테이블
    st.markdown("**과거 판매량 및 예측 수량 요약:**")
    
    # 과거 데이터 테이블
    past_summary_df = pd.DataFrame([
        {'월': month, '실제 판매량': f"{int(quantity):,}개"}
        for month, quantity in past_monthly_data.items()
    ])
    st.markdown("**과거 판매량:**")
    st.dataframe(past_summary_df, use_container_width=True)
    
    # 예측 데이터 테이블
    forecast_summary_df = pd.DataFrame([
        {'월': month, '예측 수량': f"{int(quantity):,}개"}
        for month, quantity in forecast_monthly_data.items()
    ])
    st.markdown("**향후 예측 수량:**")
    st.dataframe(forecast_summary_df, use_container_width=True)

def display_product_route_summary_chart(filtered_summary, filtered_sales, past_months, selected_product):
    """제품별 모든 경로 합계 차트 표시"""
    months = ['2025년 8월', '2025년 9월', '2025년 10월', '2025년 11월', '2025년 12월', '2026년 1월']
    
    # 선택된 제품이 있는 모든 경로 찾기
    product_routes = []
    for route, products in filtered_summary.items():
        if selected_product in products:
            product_routes.append(route)
    
    if not product_routes:
        st.warning(f"'{selected_product}' 제품에 대한 데이터가 없습니다.")
        return
    
    # 과거 판매 데이터 수집 (모든 경로 합계)
    past_monthly_data = {}
    for month in past_months:
        month_sales = 0
        for route in product_routes:
            route_sales = filtered_sales[
                (filtered_sales['경로'] == route) & 
                (filtered_sales['제품명'] == selected_product)
            ]
            month_sales += route_sales[route_sales['월'] == month]['판매수량'].sum()
        past_monthly_data[month] = month_sales
    
    # 예측 데이터 수집 (모든 경로 합계)
    forecast_monthly_data = {}
    for i, month in enumerate(months):
        month_forecast = 0
        for route in product_routes:
            if route in filtered_summary and selected_product in filtered_summary[route]:
                monthly_forecasts = filtered_summary[route][selected_product]['monthly_forecasts']
                month_forecast += int(monthly_forecasts[i]) if i < len(monthly_forecasts) else 0
        forecast_monthly_data[month] = month_forecast
    
    # 그래프 생성
    fig = go.Figure()
    
    # 과거 데이터 (실선)
    fig.add_trace(go.Scatter(
        x=list(past_monthly_data.keys()),
        y=list(past_monthly_data.values()),
        mode='lines+markers',
        name=f'{selected_product} (과거 - 모든 경로 합계)',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=8)
    ))
    
    # 예측 데이터 (점선)
    fig.add_trace(go.Scatter(
        x=list(forecast_monthly_data.keys()),
        y=list(forecast_monthly_data.values()),
        mode='lines+markers',
        name=f'{selected_product} (예측 - 모든 경로 합계)',
        line=dict(color='#1f77b4', width=3, dash='dash'),
        marker=dict(size=8, symbol='diamond')
    ))
    
    fig.update_layout(
        title=f'제품별 모든 경로 합계 판매 추이 및 향후 6개월 예측 ({selected_product})',
        xaxis_title='월',
        yaxis_title='판매/예측 수량 (개)',
        hovermode='x unified',
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 경로별 상세 정보 표시
    st.markdown(f"**경로별 상세 정보:**")
    route_info_data = []
    for route in product_routes:
        if route in filtered_summary and selected_product in filtered_summary[route]:
            info = filtered_summary[route][selected_product]
            route_info_data.append({
                '경로': route,
                '추세': info.get('trend', 'N/A'),
                '변화율': f"{info.get('change_rate', 0):.1f}%",
                '월 평균 판매량': f"{int(info.get('current_sales', 0)):,}개",
                '6개월 예측(월평균)': f"{int(info.get('total_forecast', 0)):,}개"
            })
    
    route_info_df = pd.DataFrame(route_info_data)
    st.dataframe(route_info_df, use_container_width=True)
    
    # 과거 판매량과 예측 수량 요약 테이블
    st.markdown("**과거 판매량 및 예측 수량 요약 (모든 경로 합계):**")
    
    # 과거 데이터 테이블
    past_summary_df = pd.DataFrame([
        {'월': month, '실제 판매량': f"{int(quantity):,}개"}
        for month, quantity in past_monthly_data.items()
    ])
    st.markdown("**과거 판매량:**")
    st.dataframe(past_summary_df, use_container_width=True)
    
    # 예측 데이터 테이블
    forecast_summary_df = pd.DataFrame([
        {'월': month, '예측 수량': f"{int(quantity):,}개"}
        for month, quantity in forecast_monthly_data.items()
    ])
    st.markdown("**향후 예측 수량:**")
    st.dataframe(forecast_summary_df, use_container_width=True)
    
    # 경로별 월별 예측 수량 상세 테이블
    st.markdown("**경로별 월별 예측 수량 상세:**")
    detailed_data = []
    for route in product_routes:
        if route in filtered_summary and selected_product in filtered_summary[route]:
            monthly_forecasts = filtered_summary[route][selected_product]['monthly_forecasts']
            for i, month in enumerate(months):
                detailed_data.append({
                    '경로': route,
                    '월': month,
                    '예측 수량': f"{int(monthly_forecasts[i]) if i < len(monthly_forecasts) else 0:,}개"
                })
    
    detailed_df = pd.DataFrame(detailed_data)
    st.dataframe(detailed_df, use_container_width=True)

def create_filtered_forecast_dataframe(filtered_summary):
    """필터링된 예측 결과를 데이터프레임으로 변환"""
    data = []
    months = ['2025년 8월', '2025년 9월', '2025년 10월', '2025년 11월', '2025년 12월', '2026년 1월']
    
    for route, products in filtered_summary.items():
        for product, info in products.items():
            monthly_forecasts = info['monthly_forecasts']
            
            # 원본 변화율과 보정된 변화율
            original_change_rate = info.get('original_change_rate', info['change_rate'])
            corrected_change_rate = info['change_rate']
            
            data.append({
                '경로': route,
                '제품명': product,
                '월평균_판매량': int(info['current_sales']),
                '예측_월평균_판매량': int(info['total_forecast']),
                '추세': info['trend'],
                '원본_변화율': round(original_change_rate, 1),
                '보정_변화율': round(corrected_change_rate, 1),
                '8월_예측': int(monthly_forecasts[0]) if len(monthly_forecasts) > 0 else 0,
                '9월_예측': int(monthly_forecasts[1]) if len(monthly_forecasts) > 1 else 0,
                '10월_예측': int(monthly_forecasts[2]) if len(monthly_forecasts) > 2 else 0,
                '11월_예측': int(monthly_forecasts[3]) if len(monthly_forecasts) > 3 else 0,
                '12월_예측': int(monthly_forecasts[4]) if len(monthly_forecasts) > 4 else 0,
                '1월_예측': int(monthly_forecasts[5]) if len(monthly_forecasts) > 5 else 0
            })
    
    return pd.DataFrame(data)

def show_sales_based_prediction(product_info, sales_history, kpi_history, selected_month, selected_routes):
    """
    과거 판매 데이터 기반 추세 분석 및 향후 6개월 예측
    """
    st.header("📈 판매데이터 기반 추세 분석 및 예측")
    st.markdown("---")
    
    # 동적 분석 기준 월 설정
    st.subheader("⚙️ 분석 기준 설정")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # 분석 기간 선택 (3개월, 6개월, 12개월)
        analysis_period = st.selectbox(
            "분석 기간",
            ["6개월", "3개월", "12개월"],
            index=0,
            help="과거 데이터 분석 기간을 선택하세요"
        )
    
    with col2:
        # 가중치 적용 방식 선택
        weighting_method = st.selectbox(
            "가중치 적용 방식",
            ["최근 가중", "균등 가중", "계절성 가중"],
            index=0,
            help="과거 데이터에 적용할 가중치 방식을 선택하세요"
        )
    
    with col3:
        # 보정 강도 선택
        correction_strength = st.selectbox(
            "보정 강도",
            ["보통", "강함", "약함"],
            index=0,
            help="변화율 보정의 강도를 선택하세요"
        )
    
    # 변화율 보정 설명 추가
    with st.expander("ℹ️ 동적 분석 로직 설명"):
        st.markdown(f"""
        **동적 분석 기준**:
        
        현재 선택된 분석 기간: **{analysis_period}**
        가중치 적용 방식: **{weighting_method}**
        보정 강도: **{correction_strength}**
        
                 **분석 기간별 특징**:
         - **3개월**: 최근 추세에 집중, 빠른 변화 반영
         - **6개월**: 균형잡힌 분석, 안정적인 추세 파악
         - **12개월**: 장기 추세 분석, 시간에 따른 누적 변화 반영
        
                 **가중치 적용 방식**:
         - **최근 가중**: 최근 데이터에 더 높은 가중치 적용
         - **균등 가중**: 모든 데이터에 동일한 가중치 적용
         - **계절성 가중**: 균등 가중치로 적용 (계절성 패턴 제거됨)
        
                 **보정 강도**:
         - **약함**: 작은 변화율도 그대로 반영
         - **보통**: 적당한 수준의 보정 적용
         - **강함**: 큰 변화율도 보수적으로 조정
         
         **예측 로직**:
         - **성장 추세**: 시간이 지날수록 더 성장 (매월 10%씩 가속화)
         - **하향 추세**: 시간이 지날수록 더 하향 (매월 10%씩 가속화)
         - **안정 추세**: 일정한 변화율 유지
        """)
    
    # 선택된 경로만 필터링
    filtered_product_info = product_info[product_info['경로'].isin(selected_routes)]
    
    # 분석 기준 월 설정
    analysis_month = selected_month
    
    # 전체 판매 데이터에서 선택된 경로만 필터링
    filtered_sales = sales_history[sales_history['경로'].isin(selected_routes)]
    
    # 동적 과거 월 설정
    past_months = get_dynamic_past_months(analysis_period, analysis_month)
    
    # 디버깅: 선택된 월과 분석 기간 정보 표시
    st.info(f"🔍 **분석 기준**: {analysis_month} (기준월) | {analysis_period} (분석기간) | {weighting_method} (가중치) | {correction_strength} (보정강도)")
    st.info(f"📅 **분석 대상 월**: {', '.join(past_months)}")
    
    # 선택된 월에 따른 분석 결과 미리보기
    with st.expander("🔍 분석 기준 월 상세 정보"):
        st.write(f"**선택된 기준 월**: {analysis_month}")
        st.write(f"**분석 기간**: {analysis_period}")
        st.write(f"**분석 대상 월 수**: {len(past_months)}개월")
        st.write(f"**가중치 방식**: {weighting_method}")
        st.write(f"**보정 강도**: {correction_strength}")
        
        # 월별 데이터 가용성 확인
        available_data = filtered_sales['월'].unique()
        st.write(f"**사용 가능한 데이터 월**: {', '.join(sorted(available_data))}")
        
        # 분석 대상 월과 사용 가능한 데이터 비교
        missing_months = [month for month in past_months if month not in available_data]
        if missing_months:
            st.warning(f"⚠️ **데이터 부족 월**: {', '.join(missing_months)}")
        else:
            st.success("✅ **모든 분석 대상 월에 데이터 존재**")
    
    # 동적 가중치 계산
    monthly_weights = calculate_monthly_weights(past_months, weighting_method)
    
    # 가중치 정보 표시
    weight_info = ", ".join([f"{month}: {weight:.2f}" for month, weight in monthly_weights.items()])
    st.info(f"⚖️ **월별 가중치**: {weight_info}")
    
    # 전체 예측 결과 요약 계산 (동적 파라미터 적용)
    total_forecast_summary = calculate_total_forecast_summary_dynamic(
        filtered_sales, selected_routes, past_months, monthly_weights, correction_strength
    )
    
    # 0개 판매/예측 제품 제외 및 추세별 정렬
    filtered_summary = filter_and_sort_forecast_results(total_forecast_summary)
    
    # 동적 분석 결과 요약
    st.subheader("📊 동적 분석 결과 요약")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_products = sum(len(route_data) for route_data in filtered_summary.values())
        st.metric(
            label="분석 제품 수",
            value=f"{total_products}개"
        )
    
    with col2:
        total_forecast = sum(
            sum(product_info['total_forecast'] for product_info in route_data.values())
            for route_data in filtered_summary.values()
        )
        st.metric(
            label="총 예측 수량",
            value=f"{total_forecast:,.0f}개"
        )
    
    with col3:
        avg_change_rate = sum(
            sum(product_info['change_rate'] for product_info in route_data.values())
            for route_data in filtered_summary.values()
        ) / total_products if total_products > 0 else 0
        st.metric(
            label="평균 변화율",
            value=f"{avg_change_rate:+.1f}%"
        )
    
    with col4:
        st.metric(
            label="분석 기준 월",
            value=analysis_month
        )
    
    # 제품별 판매추세 및 예측 테이블 표시
    st.subheader("📊 제품별 판매추세 및 예측")
    display_product_trend_table(filtered_summary, analysis_month)
    
    # 월별 예측 수량 추이 그래프
    st.subheader("📈 판매 추이 및 향후 6개월 예측")
    display_monthly_forecast_chart(filtered_summary, filtered_sales, past_months)
    
    # 예측 데이터 다운로드
    st.subheader("💾 예측 데이터 다운로드")
    
    forecast_df = create_filtered_forecast_dataframe(filtered_summary)
    
    csv = forecast_df.to_csv(index=False, encoding='utf-8-sig')
    st.download_button(
        label="📥 예측 결과 CSV 다운로드",
        data=csv,
        file_name=f"판매데이터_기반_예측_{analysis_month}_{analysis_period}_{weighting_method}.csv",
        mime="text/csv"
    )
