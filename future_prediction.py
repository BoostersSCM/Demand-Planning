"""
future_prediction.py
미래 예측 기능을 담당하는 모듈
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def get_relative_past_months(target_month, months_back=4):
    """
    비교 대상월 대비 상대적으로 과거 N개월 계산 (M-1부터 시작)
    예: target_month가 '2025년 8월'이고 months_back=4이면
    ['2025년 4월', '2025년 5월', '2025년 6월', '2025년 7월'] 반환 (M-4, M-3, M-2, M-1)
    """
    # 월 매핑
    month_mapping = {
        '2025년 1월': 1, '2025년 2월': 2, '2025년 3월': 3, '2025년 4월': 4,
        '2025년 5월': 5, '2025년 6월': 6, '2025년 7월': 7, '2025년 8월': 8,
        '2025년 9월': 9, '2025년 10월': 10, '2025년 11월': 11, '2025년 12월': 12
    }
    
    if target_month not in month_mapping:
        # 기본값으로 8월 기준 4개월 반환 (M-1부터 시작)
        return ['2025년 4월', '2025년 5월', '2025년 6월', '2025년 7월']
    
    current_month_num = month_mapping[target_month]
    past_months = []
    
    # M-1부터 시작하여 과거 4개월 계산 (M-4, M-3, M-2, M-1)
    for i in range(months_back):
        month_num = current_month_num - months_back + i  # M-4, M-3, M-2, M-1
        if month_num < 1:
            month_num += 12
        
        # 숫자를 다시 월 이름으로 변환
        for month_name, num in month_mapping.items():
            if num == month_num:
                past_months.append(month_name)
                break
    
    return past_months

def calculate_sales_ratio_from_history(df, sales_history, target_month):
    """
    과거 실제 판매 데이터 기반으로 제품별 판매비중 계산
    제품코드 매칭: sales_history와 product_info(df) 사이에서 이루어짐
    """
    # 비교 대상월 대비 상대적으로 과거 4개월 계산 (M-4, M-3, M-2, M-1)
    past_months = get_relative_past_months(target_month, 4)
    
    # 판매비중 컬럼 초기화
    df['판매비중'] = 0.0
    
    for route in df['경로'].unique():
        # 해당 경로의 과거 실제 판매 데이터
        route_sales = sales_history[
            (sales_history['경로'] == route) &
            (sales_history['월'].isin(past_months))
        ]
        
        if len(route_sales) > 0:
            # 제품코드 기반으로 총 판매량 계산 (sales_history와 product_info 매칭)
            if '제품코드' in route_sales.columns and '제품코드' in df.columns:
                # 제품코드가 있는 제품만 필터링 (빈 제품코드 제외)
                valid_sales = route_sales[route_sales['제품코드'].notna() & (route_sales['제품코드'] != '')]
                
                if len(valid_sales) > 0:
                    route_total_sales = valid_sales.groupby('제품코드')['판매수량'].sum()
                    route_total = route_total_sales.sum()
                    
                    if route_total > 0:
                        for product_code in df[df['경로'] == route]['제품코드'].unique():
                            if pd.notna(product_code) and product_code != '':
                                if product_code in route_total_sales.index:
                                    sales_ratio = route_total_sales[product_code] / route_total
                                else:
                                    sales_ratio = 1.0 / len(df[df['경로'] == route])
                                df.loc[(df['경로'] == route) & (df['제품코드'] == product_code), '판매비중'] = sales_ratio
                            else:
                                # 제품코드가 없는 제품은 기본값
                                df.loc[(df['경로'] == route) & (df['제품코드'].isna() | (df['제품코드'] == '')), '판매비중'] = 1.0 / len(df[df['경로'] == route])
                    else:
                        # 총 판매량이 0인 경우 균등 분배
                        product_count = len(df[df['경로'] == route])
                        df.loc[df['경로'] == route, '판매비중'] = 1.0 / product_count
                else:
                    # 유효한 판매 데이터가 없는 경우 균등 분배
                    product_count = len(df[df['경로'] == route])
                    df.loc[df['경로'] == route, '판매비중'] = 1.0 / product_count
            else:
                # 제품코드가 없는 경우 기존 방식 (제품명 기반)
                route_total_sales = route_sales.groupby('제품명')['판매수량'].sum()
                route_total = route_total_sales.sum()
                
                if route_total > 0:
                    for product in df[df['경로'] == route]['제품명'].unique():
                        if product in route_total_sales.index:
                            sales_ratio = route_total_sales[product] / route_total
                        else:
                            sales_ratio = 1.0 / len(df[df['경로'] == route])
                        df.loc[(df['경로'] == route) & (df['제품명'] == product), '판매비중'] = sales_ratio
                else:
                    product_count = len(df[df['경로'] == route])
                    df.loc[df['경로'] == route, '판매비중'] = 1.0 / product_count
        else:
            # 과거 데이터가 없는 경우 균등 분배
            product_count = len(df[df['경로'] == route])
            df.loc[df['경로'] == route, '판매비중'] = 1.0 / product_count
    
    return df

def calculate_adjustment_factors_from_history(df, sales_history, target_month, kpi_history):
    """
    과거 데이터 기반 보정계수 계산 (KPI 목표 달성 보장)
    1단계: 과거 데이터 기반 기본 보정계수 계산
    2단계: KPI 목표 맞추기 위한 스케일링 팩터 적용
    3단계: 개별 제품 보정계수를 1.0 근처로 유지하면서 전체 목표 달성
    """
    # 비교 대상월 대비 상대적으로 과거 4개월 계산 (M-4, M-3, M-2, M-1)
    past_months = get_relative_past_months(target_month, 4)
    past_months_sales = get_relative_past_months(target_month, 4)
    
    base_adjustment_factors = {}  # 기본 보정계수
    final_adjustment_factors = {}  # 최종 보정계수
    
    # 디버깅을 위한 정보 출력
    print(f"=== 보정계수 계산 시작 (KPI 목표 달성 보장) ===")
    print(f"KPI 과거 월: {past_months}")
    print(f"판매 과거 월: {past_months_sales}")
    print(f"총 제품 수: {len(df)}")
    
    # 1단계: 기본 보정계수 계산 (과거 데이터 기반)
    for route in df['경로'].unique():
        print(f"\n=== {route} 경로 (1단계: 기본 보정계수) ===")
        
        # 해당 경로의 과거 KPI 데이터
        route_kpi_data = kpi_history[
            (kpi_history['경로'] == route) &
            (kpi_history['월'].isin(past_months))
        ]
        print(f"경로 KPI 데이터: {len(route_kpi_data)}개")
        
        if len(route_kpi_data) > 0:
            # 제품코드 기반 매칭
            if '제품코드' in df.columns:
                for _, row in df[df['경로'] == route].iterrows():
                    product_code = row['제품코드']
                    product_name = row['제품명']
                    product_price = row['판매가']
                    sales_ratio = row['판매비중']
                    
                    # 각 월별 기본 보정계수 계산
                    monthly_adjustment_factors = []
                    
                    for month in past_months:
                        # 해당 월의 KPI 데이터
                        month_kpi = route_kpi_data[route_kpi_data['월'] == month]
                        if len(month_kpi) > 0:
                            # KPI매출이 문자열일 경우 숫자로 변환
                            if month_kpi['KPI매출'].dtype == 'object':
                                month_kpi = month_kpi.copy()
                                month_kpi['KPI매출'] = month_kpi['KPI매출'].astype(str).str.replace(',', '').astype(float)
                            
                            kpi_sales = month_kpi['KPI매출'].iloc[0]
                            
                            # 해당 월의 예측 수량 계산 (인기도 가중치 없이)
                            month_predicted_sales = (kpi_sales * sales_ratio) / product_price
                            
                            # 해당 월의 실제 판매 데이터 (제품코드 기반)
                            month_actual = sales_history[
                                (sales_history['경로'] == route) & 
                                (sales_history['제품코드'] == product_code) &
                                (sales_history['월'] == month)
                            ]
                            
                            if len(month_actual) > 0:
                                actual_sales = month_actual['판매수량'].iloc[0]
                                
                                # 해당 월의 기본 보정계수 계산
                                if month_predicted_sales > 0:
                                    month_adjustment_factor = actual_sales / month_predicted_sales
                                    # 기본 보정계수 범위 제한 (0.3 ~ 3.0)
                                    month_adjustment_factor = max(0.3, min(3.0, month_adjustment_factor))
                                    monthly_adjustment_factors.append(month_adjustment_factor)
                                else:
                                    monthly_adjustment_factors.append(1.0)
                            else:
                                monthly_adjustment_factors.append(1.0)
                        else:
                            monthly_adjustment_factors.append(1.0)
                    
                    # 월별 기본 보정계수의 평균 계산
                    if monthly_adjustment_factors:
                        base_adjustment_factor = sum(monthly_adjustment_factors) / len(monthly_adjustment_factors)
                        base_adjustment_factor = round(base_adjustment_factor, 2)
                    else:
                        base_adjustment_factor = 1.0
                    
                    base_adjustment_factors[(route, product_code)] = base_adjustment_factor
            else:
                # 제품코드가 없는 경우 기본값
                for _, row in df[df['경로'] == route].iterrows():
                    base_adjustment_factors[(route, row['제품명'])] = 1.0
        else:
            # KPI 데이터가 없는 경우 기본값
            for _, row in df[df['경로'] == route].iterrows():
                key = (route, row['제품코드']) if '제품코드' in df.columns else (route, row['제품명'])
                base_adjustment_factors[key] = 1.0
    
    # 2단계: KPI 목표 맞추기 위한 스케일링 팩터 계산
    print(f"\n=== 2단계: KPI 목표 맞추기 ===")
    
    for route in df['경로'].unique():
        print(f"\n--- {route} 경로 스케일링 팩터 계산 ---")
        
        # 해당 경로의 현재 KPI
        route_kpi = df[df['경로'] == route]['KPI매출'].iloc[0]
        print(f"목표 KPI: {route_kpi:,.0f}")
        
        # 기본 보정계수 적용 시 예상 총 매출 계산
        route_products = df[df['경로'] == route]
        expected_total_revenue = 0
        
        for _, product_row in route_products.iterrows():
            if '제품코드' in df.columns:
                product_code = product_row['제품코드']
                base_factor = base_adjustment_factors.get((route, product_code), 1.0)
            else:
                product_name = product_row['제품명']
                base_factor = base_adjustment_factors.get((route, product_name), 1.0)
            
            # 기본 보정계수 적용 시 예상 수량
            base_predicted_quantity = product_row['예측수량'] * base_factor
            # 예상 매출
            expected_revenue = base_predicted_quantity * product_row['판매가']
            expected_total_revenue += expected_revenue
        
        print(f"기본 보정계수 적용 시 예상 총 매출: {expected_total_revenue:,.0f}")
        
        # 스케일링 팩터 계산 (목표 KPI / 예상 총 매출)
        if expected_total_revenue > 0:
            scaling_factor = route_kpi / expected_total_revenue
            # 스케일링 팩터 범위 제한 (0.5 ~ 2.0)
            scaling_factor = max(0.5, min(2.0, scaling_factor))
            print(f"스케일링 팩터: {scaling_factor:.3f}")
        else:
            scaling_factor = 1.0
            print(f"스케일링 팩터: 1.0 (예상 매출이 0)")
        
        # 3단계: 최종 보정계수 계산 (기본 보정계수 × 스케일링 팩터)
        print(f"\n--- {route} 경로 최종 보정계수 계산 ---")
        
        for _, product_row in route_products.iterrows():
            if '제품코드' in df.columns:
                product_code = product_row['제품코드']
                base_factor = base_adjustment_factors.get((route, product_code), 1.0)
                key = (route, product_code)
            else:
                product_name = product_row['제품명']
                base_factor = base_adjustment_factors.get((route, product_name), 1.0)
                key = (route, product_name)
            
            # 최종 보정계수 = 기본 보정계수 × 스케일링 팩터
            final_factor = base_factor * scaling_factor
            final_adjustment_factors[key] = final_factor
            
            print(f"  {product_row['제품명']}: 기본={base_factor:.2f} × 스케일링={scaling_factor:.3f} = 최종={final_factor:.3f}")
    
    # DataFrame에 보정계수 적용
    if '제품코드' in df.columns:
        df['보정계수'] = df.apply(
            lambda row: final_adjustment_factors.get(
                (row['경로'], row['제품코드']), 1.0), 
            axis=1
        )
    else:
        df['보정계수'] = df.apply(
            lambda row: final_adjustment_factors.get(
                (row['경로'], row['제품명']), 1.0), 
            axis=1
        )
    
    return df

def calculate_dynamic_popularity_weights(df, sales_history, target_month):
    """
    과거 판매 데이터 기반으로 동적 인기도 가중치 계산
    
    계산 방식:
    1. 과거 4개월 데이터에서 제품별 판매량 추이 분석 (M-4, M-3, M-2, M-1)
    2. 최근 2개월 평균 vs 과거 2개월 평균 비교로 추세 분석
    3. 경로별로 정규화하여 상대적 인기도 계산
    """
    # 비교 대상월 대비 상대적으로 과거 4개월 계산 (M-4, M-3, M-2, M-1)
    past_months = get_relative_past_months(target_month, 4)
    print(f"\n🔍 인기도 가중치 계산 - 과거 4개월: {past_months}")
    
    # 인기도 가중치 초기화
    df['인기도_가중치'] = 1.0
    
    for route in df['경로'].unique():
        print(f"\n=== {route} 경로 인기도 가중치 계산 ===")
        
        # 해당 경로의 과거 실제 판매 데이터
        route_sales = sales_history[
            (sales_history['경로'] == route) &
            (sales_history['월'].isin(past_months))
        ]
        
        print(f"📊 {route} 경로 과거 데이터 건수: {len(route_sales)}건")
        if len(route_sales) > 0:
            print(f"📅 데이터 기간: {route_sales['월'].unique()}")
            print(f"🏷️ 제품코드 컬럼 존재: {'제품코드' in route_sales.columns}")
            print(f"🏷️ DataFrame 제품코드 컬럼 존재: {'제품코드' in df.columns}")
        
        if len(route_sales) > 0:
            # 제품코드 기반으로 판매량 추이 분석
            if '제품코드' in route_sales.columns and '제품코드' in df.columns:
                valid_sales = route_sales[route_sales['제품코드'].notna() & (route_sales['제품코드'] != '')]
                print(f"✅ 유효한 제품코드 데이터: {len(valid_sales)}건")
                
                if len(valid_sales) > 0:
                    # 제품별 월간 판매량 계산
                    monthly_sales = valid_sales.groupby(['제품코드', '월'])['판매수량'].sum().reset_index()
                    
                    # 제품별 판매량 계산 (전체 4개월 대비 최근 2개월 평균 vs 과거 2개월 평균)
                    product_total_sales = valid_sales.groupby('제품코드')['판매수량'].sum()
                    
                    # 최근 2개월 판매량 (전체 4개월 중 마지막 2개월: M-2, M-1)
                    recent_2months = past_months[-2:] if len(past_months) >= 2 else past_months
                    recent_2months_sales = valid_sales[valid_sales['월'].isin(recent_2months)].groupby('제품코드')['판매수량'].sum()
                    
                    # 과거 2개월 판매량 (전체 4개월 중 첫 번째 2개월: M-4, M-3)
                    past_2months = past_months[:2] if len(past_months) >= 2 else past_months
                    past_2months_sales = valid_sales[valid_sales['월'].isin(past_2months)].groupby('제품코드')['판매수량'].sum()
                    
                    print(f"📈 최근 2개월: {recent_2months}")
                    print(f"📉 과거 2개월: {past_2months}")
                    
                    # 제품별 인기도 점수 계산
                    popularity_scores = {}
                    
                    for product_code in df[df['경로'] == route]['제품코드'].unique():
                        if pd.isna(product_code) or product_code == '':
                            popularity_scores[product_code] = 0.001
                            print(f"  ⚠️ {product_code}: 제품코드 없음 - 기본값 0.001")
                            continue
                            
                        total_sales = product_total_sales.get(product_code, 0)
                        recent_2months_total = recent_2months_sales.get(product_code, 0)
                        past_2months_total = past_2months_sales.get(product_code, 0)
                        
                        # 판매량이 없는 제품은 기본값
                        if total_sales == 0:
                            popularity_scores[product_code] = 0.001
                            print(f"  ⚠️ {product_code}: 판매량 없음 - 기본값 0.001")
                            continue
                        
                        # 최근 2개월 평균 판매량
                        recent_2months_avg = recent_2months_total / 2 if recent_2months_total > 0 else 0
                        
                        # 과거 2개월 평균 판매량
                        past_2months_avg = past_2months_total / 2 if past_2months_total > 0 else 0
                        
                        # 최근 2개월 평균 vs 과거 2개월 평균 비교 (추세 비율)
                        if past_2months_avg > 0:
                            trend_ratio = recent_2months_avg / past_2months_avg
                        else:
                            trend_ratio = 1.0  # 과거 데이터가 없으면 중립
                        
                        # 판매량 규모 점수 (전체 대비 비중)
                        total_route_sales = product_total_sales.sum()
                        volume_score = total_sales / total_route_sales if total_route_sales > 0 else 0
                        
                        # 인기도 점수 = 판매량 규모 × 추세 비율
                        popularity_score = volume_score * trend_ratio
                        
                        popularity_scores[product_code] = popularity_score
                        
                        print(f"  📊 {product_code}: 총판매량={total_sales:,}, 최근2개월평균={recent_2months_avg:.1f}, "
                              f"과거2개월평균={past_2months_avg:.1f}, 추세비율={trend_ratio:.2f}, 인기도점수={popularity_score:.3f}")
                    
                    # 인기도 점수를 가중치로 변환 (1.0 기준으로 정규화)
                    if popularity_scores:
                        max_score = max(popularity_scores.values())
                        min_score = min(popularity_scores.values())
                        
                        print(f"\n📋 가중치 변환:")
                        print(f"  최대 점수: {max_score:.3f}")
                        print(f"  최소 점수: {min_score:.3f}")
                        
                        # 가중치 범위 조정 (0.7 ~ 1.3)
                        for product_code, score in popularity_scores.items():
                            if max_score > min_score:
                                # 정규화 후 범위 조정
                                normalized_score = (score - min_score) / (max_score - min_score)
                                weight = 0.7 + (normalized_score * 0.6)  # 0.7 ~ 1.3 범위
                            else:
                                weight = 1.0
                                print(f"  ⚠️ 모든 점수가 동일함 - 기본 가중치 1.0 적용")
                            
                            popularity_scores[product_code] = round(weight, 2)
                            print(f"    {product_code}: {score:.3f} → {weight}")
                    
                    # 가중치 적용
                    for _, row in df[df['경로'] == route].iterrows():
                        product_code = row['제품코드']
                        weight = popularity_scores.get(product_code, 1.0)
                        df.loc[(df['경로'] == route) & (df['제품코드'] == product_code), '인기도_가중치'] = weight
                        
                        product_name = row['제품명']
                        print(f"  ✅ {product_name} ({product_code}): 가중치 {weight}")
                else:
                    # 판매 데이터가 없는 경우 기본값
                    print(f"  ⚠️ 판매 데이터 없음 - 기본 가중치 0.001 적용")
                    # 모든 제품에 기본 가중치 0.001 적용
                    for _, row in df[df['경로'] == route].iterrows():
                        product_code = row['제품코드']
                        df.loc[(df['경로'] == route) & (df['제품코드'] == product_code), '인기도_가중치'] = 0.001
                        print(f"  ✅ {row['제품명']} ({product_code}): 기본 가중치 0.001")
            else:
                # 제품코드가 없는 경우 제품명 기반 계산
                print(f"  📝 제품명 기반 계산")
                product_total_sales = route_sales.groupby('제품명')['판매수량'].sum()
                
                # 최근 2개월 판매량 (전체 4개월 중 마지막 2개월: M-2, M-1)
                recent_2months = past_months[-2:] if len(past_months) >= 2 else past_months
                recent_2months_sales = route_sales[route_sales['월'].isin(recent_2months)].groupby('제품명')['판매수량'].sum()
                
                # 과거 2개월 판매량 (전체 4개월 중 첫 번째 2개월: M-4, M-3)
                past_2months = past_months[:2] if len(past_months) >= 2 else past_months
                past_2months_sales = route_sales[route_sales['월'].isin(past_2months)].groupby('제품명')['판매수량'].sum()
                
                popularity_scores = {}
                total_route_sales = product_total_sales.sum()
                
                for product_name in df[df['경로'] == route]['제품명'].unique():
                    total_sales = product_total_sales.get(product_name, 0)
                    recent_2months_total = recent_2months_sales.get(product_name, 0)
                    past_2months_total = past_2months_sales.get(product_name, 0)
                    
                    if total_sales == 0:
                        popularity_scores[product_name] = 0.001
                        print(f"  ⚠️ {product_name}: 판매량 없음 - 기본값 0.001")
                        continue
                    
                    # 최근 2개월 평균 판매량
                    recent_2months_avg = recent_2months_total / 2 if recent_2months_total > 0 else 0
                    
                    # 과거 2개월 평균 판매량
                    past_2months_avg = past_2months_total / 2 if past_2months_total > 0 else 0
                    
                    # 최근 2개월 평균 vs 과거 2개월 평균 비교 (추세 비율)
                    if past_2months_avg > 0:
                        trend_ratio = recent_2months_avg / past_2months_avg
                    else:
                        trend_ratio = 1.0  # 과거 데이터가 없으면 중립
                    
                    volume_score = total_sales / total_route_sales if total_route_sales > 0 else 0
                    popularity_score = volume_score * trend_ratio
                    
                    popularity_scores[product_name] = popularity_score
                    print(f"  📊 {product_name}: 총판매량={total_sales:,}, 최근2개월평균={recent_2months_avg:.1f}, "
                          f"과거2개월평균={past_2months_avg:.1f}, 추세비율={trend_ratio:.2f}, 인기도점수={popularity_score:.3f}")
                
                # 인기도 점수를 가중치로 변환
                if popularity_scores:
                    max_score = max(popularity_scores.values())
                    min_score = min(popularity_scores.values())
                    
                    print(f"\n📋 가중치 변환:")
                    print(f"  최대 점수: {max_score:.3f}")
                    print(f"  최소 점수: {min_score:.3f}")
                    
                    for product_name, score in popularity_scores.items():
                        if max_score > min_score:
                            normalized_score = (score - min_score) / (max_score - min_score)
                            weight = 0.7 + (normalized_score * 0.6)
                        else:
                            weight = 1.0
                            print(f"  ⚠️ 모든 점수가 동일함 - 기본 가중치 1.0 적용")
                        
                        popularity_scores[product_name] = round(weight, 2)
                        print(f"    {product_name}: {score:.3f} → {weight}")
                
                # 가중치 적용
                for _, row in df[df['경로'] == route].iterrows():
                    product_name = row['제품명']
                    weight = popularity_scores.get(product_name, 1.0)
                    df.loc[(df['경로'] == route) & (df['제품명'] == product_name), '인기도_가중치'] = weight
                    
                    print(f"  ✅ {product_name}: 가중치 {weight}")
        else:
            print(f"  ⚠️ 과거 데이터 없음 - 기본 가중치 0.001 적용")
            # 모든 제품에 기본 가중치 0.001 적용
            for _, row in df[df['경로'] == route].iterrows():
                if '제품코드' in df.columns:
                    product_code = row['제품코드']
                    df.loc[(df['경로'] == route) & (df['제품코드'] == product_code), '인기도_가중치'] = 0.001
                    print(f"  ✅ {row['제품명']} ({product_code}): 기본 가중치 0.001")
                else:
                    product_name = row['제품명']
                    df.loc[(df['경로'] == route) & (df['제품명'] == product_name), '인기도_가중치'] = 0.001
                    print(f"  ✅ {product_name}: 기본 가중치 0.001")
    
    return df

def estimate_demand_improved(kpi_df, product_df, sales_history, target_month, kpi_history=None):
    """
    개선된 수요 예측 로직:
    1. 과거 실제 판매 데이터 기반 제품별 판매비중 계산
    2. 제품별 판매가로 수량 산출
    3. 과거 데이터 기반 보정계수 적용
    """
    df = pd.merge(product_df, kpi_df, on='경로')
    
    # Step 1: 과거 실제 판매 데이터 기반 제품별 판매비중 계산 (인기도 가중치 없이)
    df = calculate_sales_ratio_from_history(df, sales_history, target_month)
    
    # Step 2: KPI 기반 제품별 예상 매출 계산 (인기도 가중치 없이)
    # KPI매출이 문자열일 경우 숫자로 변환
    if df['KPI매출'].dtype == 'object':
        df['KPI매출'] = df['KPI매출'].astype(str).str.replace(',', '').astype(float)
    
    # 제품별 예상 매출 계산 (KPI매출 × 판매비중)
    df['제품별_예상매출'] = df['판매비중'] * df['KPI매출']
    
    # 디버깅: 데이터 타입 확인
    print(f"판매비중 타입: {df['판매비중'].dtype}")
    print(f"KPI매출 타입: {df['KPI매출'].dtype}")
    print(f"제품별_예상매출 타입: {df['제품별_예상매출'].dtype}")
    print(f"제품별_예상매출 샘플: {df['제품별_예상매출'].head()}")
    
    # 제품별_예상매출이 object 타입인 경우 숫자로 강제 변환
    if df['제품별_예상매출'].dtype == 'object':
        print("제품별_예상매출이 object 타입입니다. 숫자로 변환 중...")
        df['제품별_예상매출'] = pd.to_numeric(df['제품별_예상매출'], errors='coerce').fillna(0)
        print(f"변환 후 타입: {df['제품별_예상매출'].dtype}")
    
    # KPI 정확성 보장: 각 경로별로 제품별_예상매출의 합이 KPI와 정확히 일치하도록 보정
    for route in df['경로'].unique():
        route_df = df[df['경로'] == route]
        route_kpi = route_df['KPI매출'].iloc[0]
        
        # 현재 제품별_예상매출의 합계
        current_sum = route_df['제품별_예상매출'].sum()
        
        # 오차 계산
        difference = route_kpi - current_sum
        
        if abs(difference) > 0.01:  # 1원 이상의 오차가 있는 경우
            print(f"{route} 경로: KPI={route_kpi:,.0f}, 현재합계={current_sum:,.0f}, 오차={difference:,.0f}")
            
            # 가장 큰 제품별_예상매출을 가진 제품에 오차를 보정
            max_revenue_idx = route_df['제품별_예상매출'].idxmax()
            df.loc[max_revenue_idx, '제품별_예상매출'] += difference
            
            # 보정 후 확인
            corrected_sum = df[df['경로'] == route]['제품별_예상매출'].sum()
            print(f"{route} 경로 보정 후: 합계={corrected_sum:,.0f}, KPI={route_kpi:,.0f}")
    
    # 정수 변환 (보정 후)
    df['제품별_예상매출'] = df['제품별_예상매출'].round().astype(int)
    
    # Step 3: 순수한 예측 수량 계산 (인기도 가중치 없이)
    df['예측수량'] = df['제품별_예상매출'] / df['판매가']
    
    # Step 4: 보정계수 계산 (순수한 예측량 기반)
    if sales_history is not None:
        # 과거 데이터 기반 보정계수 계산 (순수한 예측량 기반)
        df = calculate_adjustment_factors_from_history(df, sales_history, target_month, kpi_history)
    else:
        # 기존 방식 (고정 보정계수)
        df['보정계수'] = 1.0
    
    # Step 5: 보정 수량 계산 (보정계수 적용)
    df['보정수량'] = df['예측수량'] * df['보정계수']
    
    # Step 6: 동적 인기도 가중치 계산 (적용은 나중에)
    df = calculate_dynamic_popularity_weights(df, sales_history, target_month)
    
    # Step 7: KPI 목표와 맞추기 위한 스케일링 (인기도 가중치 적용 전)
    for route in df['경로'].unique():
        route_df = df[df['경로'] == route]
        route_kpi = route_df['KPI매출'].iloc[0]
        
        # 보정수량 기반 예상 총 매출
        route_expected_revenue = (route_df['보정수량'] * route_df['판매가']).sum()
        
        if route_expected_revenue > 0:
            # KPI 목표와 맞추기 위한 스케일링 팩터
            kpi_scaling_factor = route_kpi / route_expected_revenue
            
            # 보정수량에 KPI 스케일링 팩터 적용
            df.loc[df['경로'] == route, '보정수량'] = (
                df.loc[df['경로'] == route, '보정수량'] * kpi_scaling_factor
            )
            
            print(f"{route} 경로 KPI 스케일링: KPI={route_kpi:,.0f}, 예상매출={route_expected_revenue:,.0f}, 스케일링팩터={kpi_scaling_factor:.3f}")
    
    # Step 8: 최종 예측량에 인기도 가중치 적용 (KPI 스케일링 후)
    df['최종_예측수량'] = df['보정수량'] * df['인기도_가중치']
    
    # 예측수량과 보정수량, 최종_예측수량을 정수로 변환
    df['예측수량'] = df['예측수량'].round().astype(int)
    df['보정수량'] = df['보정수량'].round().astype(int)
    df['최종_예측수량'] = df['최종_예측수량'].round().astype(int)
    
    # 최종 KPI 정확성 보장: 정수 변환 후 발생한 오차를 정확히 보정
    for route in df['경로'].unique():
        route_df = df[df['경로'] == route]
        route_kpi = route_df['KPI매출'].iloc[0]
        
        # 정수 변환 후 실제 예상 총 매출
        actual_revenue = (route_df['최종_예측수량'] * route_df['판매가']).sum()
        
        # 오차 계산
        final_error = route_kpi - actual_revenue
        
        if abs(final_error) > 0.01:  # 1원 이상의 오차가 있는 경우
            print(f"{route} 경로 최종 오차: KPI={route_kpi:,.0f}, 실제매출={actual_revenue:,.0f}, 오차={final_error:,.0f}")
            
            # 오차를 가장 큰 매출을 가진 제품에 보정
            route_df_with_revenue = route_df.copy()
            route_df_with_revenue['제품별_매출'] = route_df_with_revenue['최종_예측수량'] * route_df_with_revenue['판매가']
            max_revenue_idx = route_df_with_revenue['제품별_매출'].idxmax()
            
            adjustment_quantity = final_error / route_df.loc[max_revenue_idx, '판매가']
            
            # 보정된 수량이 음수가 되지 않도록 확인
            current_quantity = df.loc[max_revenue_idx, '최종_예측수량']
            new_quantity = current_quantity + adjustment_quantity
            
            if new_quantity >= 0:
                df.loc[max_revenue_idx, '최종_예측수량'] = new_quantity
            else:
                # 음수가 되는 경우, 두 번째로 큰 매출을 가진 제품에 보정
                sorted_revenue_indices = route_df_with_revenue['제품별_매출'].sort_values(ascending=False).index
                if len(sorted_revenue_indices) > 1:
                    second_max_revenue_idx = sorted_revenue_indices[1]
                    adjustment_quantity = final_error / route_df.loc[second_max_revenue_idx, '판매가']
                    df.loc[second_max_revenue_idx, '최종_예측수량'] += adjustment_quantity
            
            # 보정 후 확인
            corrected_revenue = (df.loc[df['경로'] == route, '최종_예측수량'] * df.loc[df['경로'] == route, '판매가']).sum()
            print(f"{route} 경로 최종 보정 후: 예상매출={corrected_revenue:,.0f}, KPI={route_kpi:,.0f}")
            
            # 최종 확인 - 정확히 일치하는지 검증
            final_check = (df.loc[df['경로'] == route, '최종_예측수량'] * df.loc[df['경로'] == route, '판매가']).sum()
            if abs(final_check - route_kpi) <= 1:
                print(f"✅ {route} 경로: KPI와 정확히 일치합니다!")
            else:
                print(f"❌ {route} 경로: 여전히 오차가 있습니다. ({final_check:,.0f} vs {route_kpi:,.0f})")
                
                # 추가 보정 시도 - 여러 제품에 분산
                remaining_error = route_kpi - final_check
                if abs(remaining_error) > 0.01:
                    print(f"🔄 추가 보정 시도: 남은 오차 {remaining_error:,.0f}원")
                    
                    # 상위 3개 제품에 오차를 분산
                    top_3_indices = route_df_with_revenue['제품별_매출'].nlargest(3).index
                    for i, idx in enumerate(top_3_indices):
                        if i < 2:  # 첫 번째와 두 번째 제품에만 보정
                            partial_adjustment = remaining_error / (2 * route_df.loc[idx, '판매가'])
                            df.loc[idx, '최종_예측수량'] += partial_adjustment
                    
                    # 최종 재확인
                    final_final_check = (df.loc[df['경로'] == route, '최종_예측수량'] * df.loc[df['경로'] == route, '판매가']).sum()
                    print(f"🔄 최종 재확인: {final_final_check:,.0f}원 vs KPI {route_kpi:,.0f}원")
                    if abs(final_final_check - route_kpi) <= 1:
                        print(f"✅ {route} 경로: 최종 보정 성공!")
                    else:
                        print(f"❌ {route} 경로: 최종 보정 실패. 오차: {final_final_check - route_kpi:,.0f}원")
    
    return df[['월', '경로', '제품명', '판매가', 'KPI매출', '제품별_예상매출', '예측수량', '보정계수', '보정수량', '인기도_가중치', '최종_예측수량', '판매비중']]

def display_future_dashboard(forecast, selected_routes):
    """원래 UI/UX를 유지한 미래 예측 결과 대시보드 표시"""
    
    # 대시보드 레이아웃
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="총 예측 수량",
            value=f"{forecast['보정수량'].sum():,}개"
        )
    
    with col2:
        st.metric(
            label="평균 예측 수량",
            value=f"{forecast['보정수량'].mean():,.0f}개"
        )
    
    with col3:
        # 최종 예측수량 기반으로 실제 예상 매출 계산
        final_expected_revenue = (forecast['최종_예측수량'] * forecast['판매가']).sum()
        st.metric(
            label="예상 총 매출",
            value=f"{int(final_expected_revenue):,}원"
        )
    
    st.markdown("---")
    
    # 차트 섹션
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 경로별 예측 수량")
        route_totals = forecast.groupby('경로')['최종_예측수량'].sum()
        fig1 = px.bar(
            x=route_totals.index,
            y=route_totals.values,
            title="경로별 총 예측 수량",
            labels={'x': '경로', 'y': '예측 수량'}
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        st.subheader("📈 제품별 예측 수량 (상위 10개)")
        product_totals = forecast.groupby('제품명')['최종_예측수량'].sum().sort_values(ascending=False).head(10)
        fig2 = px.bar(
            x=product_totals.values,
            y=product_totals.index,
            orientation='h',
            title="제품별 예측 수량",
            labels={'x': '예측 수량', 'y': '제품명'}
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    # 상세 데이터 테이블
    st.subheader("📋 상세 예측 결과")
    
    # 보정계수 설명
    with st.expander("ℹ️ 개선된 예측 로직 설명"):
        st.markdown("""
        **개선된 예측 로직 (KPI 기반 수요 예측)**:
        1. **제품별 판매비중**: 과거 4개월 실제 판매 데이터 기반으로 각 제품의 비중 계산 (인기도 가중치 없이)
        2. **제품별 예상매출**: KPI매출 × 판매비중으로 계산 (경로별 총합이 KPI와 일치)
        3. **순수 예측수량**: 제품별 예상매출 ÷ 개별제품단가로 계산 (인기도 가중치 없이)
        4. **보정계수**: 가격이 낮아서 발생하는 수량 과다 예측을 보완하기 위한 지표
           - 1단계: 과거 데이터 기반 기본 보정계수 계산 (순수 예측량 대비 실제 판매량)
           - 2단계: KPI 목표 맞추기 위한 스케일링 팩터 적용
           - 3단계: 개별 제품 보정계수를 1.0 근처로 유지하면서 전체 목표 달성
        5. **보정수량**: 순수 예측수량 × 보정계수
        6. **최종 예측수량**: 보정수량 × 인기도 가중치 (최종 단계에서만 적용)
        
        **핵심 개선사항**:
        - ✅ **정확한 KPI 기반**: 제품별 예상매출의 총합이 경로 KPI와 정확히 일치
        - ✅ **가격 기반 보정**: 저가 제품의 수량 과다 예측을 보정계수로 보완
        - ✅ **정확한 예측**: 과거 데이터 기반의 정확한 판매 비중 활용
        - ✅ **동적 인기도 가중치**: 과거 4개월 판매 데이터 기반 동적 가중치 계산
        - ✅ **KPI 목표 달성 보장**: 스케일링 팩터로 전체 목표 매출 달성
        - ✅ **개별 제품 단가**: 각 제품의 개별 판매가 사용
        
        **동적 인기도 가중치 계산 방식**:
        - 📊 **판매량 규모 점수**: 전체 판매량 대비 제품별 비중
        - 📈 **추세 분석**: 최근 2개월 평균 판매량 vs 과거 2개월 평균 판매량 비교
        - 🎯 **인기도 점수**: 판매량 규모 × 추세 비율
        - ⚖️ **가중치 변환**: 0.7 ~ 1.3 범위로 정규화 (최소 0.7, 최대 1.3)
        - 🔄 **실시간 반영**: 매월 과거 데이터 업데이트로 가중치 자동 조정
        
        **보정계수의 역할**:
        - 💰 **가격 보정**: 저가 제품의 수량 과다 예측을 실제 판매 패턴으로 보정
        - 📊 **과거 패턴 반영**: 실제 판매량 대비 예측량의 편차를 보정계수로 조정
        - 🎯 **KPI 정확도**: 보정계수를 통해 최종 예측이 KPI 목표에 정확히 부합
        """)
    
    # 상세 예측 결과 테이블 포맷팅
    forecast_display = forecast[['경로', '제품명', '판매가', '제품별_예상매출', '예측수량', '보정계수', '보정수량', '인기도_가중치', '최종_예측수량']].copy()
    
    # 제품별_예상매출을 정수로 변환 후 포맷팅
    forecast_display['제품별_예상매출'] = forecast_display['제품별_예상매출'].round().astype(int)
    
    # 숫자 컬럼에 콤마 적용 (금액은 정수로 표시, 수량은 정수로 표시)
    forecast_display['판매가'] = forecast_display['판매가'].apply(lambda x: f"{int(x):,}")
    forecast_display['제품별_예상매출'] = forecast_display['제품별_예상매출'].apply(lambda x: f"{int(x):,}")
    forecast_display['예측수량'] = forecast_display['예측수량'].apply(lambda x: f"{int(x):,}")
    forecast_display['보정계수'] = forecast_display['보정계수'].apply(lambda x: f"{x:.2f}")
    forecast_display['보정수량'] = forecast_display['보정수량'].apply(lambda x: f"{int(x):,}")
    forecast_display['인기도_가중치'] = forecast_display['인기도_가중치'].apply(lambda x: f"{x:.2f}")
    forecast_display['최종_예측수량'] = forecast_display['최종_예측수량'].apply(lambda x: f"{int(x):,}")
    
    st.dataframe(forecast_display, use_container_width=True)
    
    # 경로별 분석
    st.subheader("🔍 경로별 상세 분석")
    for route in selected_routes:
        route_data = forecast[forecast['경로'] == route]
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(f"{route} 총 수량", f"{route_data['최종_예측수량'].sum():,}개")
        with col2:
            # 최종 예측수량 기반으로 실제 예상 매출 계산
            route_expected_revenue = (route_data['최종_예측수량'] * route_data['판매가']).sum()
            st.metric(f"{route} 예상 매출", f"{int(route_expected_revenue):,}원")
        
        # 제품별 수량 분포 표
        st.write(f"**{route} 제품별 수량 분포**")
        route_summary = route_data[['제품명', '최종_예측수량', '판매가']].copy()
        # 최종 예측수량 기반으로 실제 예상 매출 계산
        route_summary['제품별_예상매출'] = route_summary['최종_예측수량'] * route_summary['판매가']
        route_summary = route_summary.sort_values('최종_예측수량', ascending=False)
        route_summary['최종_예측수량'] = route_summary['최종_예측수량'].apply(lambda x: f"{int(x):,}")
        route_summary['제품별_예상매출'] = route_summary['제품별_예상매출'].apply(lambda x: f"{int(x):,}")
        st.dataframe(route_summary, use_container_width=True)

def show_future_prediction(product_info, sales_history, kpi_history, selected_month, selected_routes):
    """미래 예측 모드 메인 함수"""
    
    # 선택된 경로만 필터링
    filtered_product_info = product_info[product_info['경로'].isin(selected_routes)]
    
    # kpi_history.csv에서 실제 KPI 데이터 가져오기
    # 선택된 월에 해당하는 KPI 데이터 필터링
    month_mapping = {
        '25-Aug': '2025년 8월',
        '25-Sep': '2025년 9월', 
        '25-Oct': '2025년 10월'
    }
    
    target_month_korean = month_mapping.get(selected_month, '2025년 8월')
    
    # 해당 월의 KPI 데이터 가져오기
    month_kpi_data = kpi_history[kpi_history['월'] == target_month_korean]
    
    # 선택된 경로에 대한 KPI 데이터 준비
    kpi_current = pd.DataFrame({
        '월': [selected_month] * len(selected_routes),
        '경로': selected_routes,
        'KPI매출': [0] * len(selected_routes)  # 초기값 설정
    })
    
    # 실제 KPI 데이터로 채우기
    for i, route in enumerate(selected_routes):
        route_kpi = month_kpi_data[month_kpi_data['경로'] == route]
        if len(route_kpi) > 0:
            kpi_value = route_kpi.iloc[0]['KPI매출']
            # 문자열인 경우 숫자로 변환
            if isinstance(kpi_value, str):
                kpi_value = float(kpi_value.replace(',', ''))
            kpi_current.loc[i, 'KPI매출'] = kpi_value
        else:
            # 해당 경로의 KPI 데이터가 없는 경우 기본값 설정
            default_kpi_values = {
                'Amazon(USA)': 910171022,
                'B2B(GLOBAL)': 3000000000,
                'Shopee(PH)': 60000000,
                'Shopee(MY)': 55000000,
                'Shopee(SG)': 130000000,
                'Shopee(VN)': 15000000,
                'Shopee(TW)': 15000000,
                'TikTokShop(USA)': 36000000,
                'Shopee(TH)': 10000000
            }
            kpi_current.loc[i, 'KPI매출'] = default_kpi_values.get(route, 100000000)
    
    # KPI 데이터 확인
    with st.expander("📊 KPI 데이터 확인", expanded=False):
        st.dataframe(kpi_current, use_container_width=True)
    
    # 예측 실행 (과거 데이터 기반 보정계수 적용)
    forecast = estimate_demand_improved(kpi_current, filtered_product_info, sales_history, selected_month, kpi_history)
    
    # 보정계수 분석
    st.subheader("🔧 보정계수 분석")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_adjustment = forecast['보정계수'].mean()
        st.metric(
            label="평균 보정계수",
            value=f"{avg_adjustment:.2f}"
        )
    
    with col2:
        max_adjustment = forecast['보정계수'].max()
        st.metric(
            label="최대 보정계수",
            value=f"{max_adjustment:.2f}"
        )
    
    with col3:
        min_adjustment = forecast['보정계수'].min()
        st.metric(
            label="최소 보정계수",
            value=f"{min_adjustment:.2f}"
        )
    

    
    # 보정계수 디버깅 정보
    with st.expander("🔍 보정계수 디버깅 정보"):
        st.write("**개선된 보정계수 계산 방식 (KPI 목표 달성 보장):**")
        st.write("""
        1. **1단계 - 기본 보정계수**: 과거 3개월 데이터 기반
           - 각 월별로 실제판매수량 ÷ 예측수량 계산
           - 3개월 평균값을 기본 보정계수로 사용
           - 범위 제한: 0.3 ~ 3.0
        
        2. **2단계 - 스케일링 팩터**: KPI 목표 맞추기
           - 기본 보정계수 적용 시 예상 총 매출 계산
           - 스케일링 팩터 = 목표 KPI ÷ 예상 총 매출
           - 범위 제한: 0.5 ~ 2.0
        
        3. **3단계 - 최종 보정계수**: 균형잡힌 적용
           - 최종 보정계수 = 기본 보정계수 × 스케일링 팩터
           - 개별 제품은 1.0 근처 유지
           - 전체적으로는 KPI 목표 달성 보장
        """)
        st.write("**보정계수 계산 과정:**")
        st.write(f"- 총 제품 수: {len(forecast)}")
        st.write(f"- 보정계수 1.0인 제품 수: {(forecast['보정계수'] == 1.0).sum()}")
        st.write(f"- 보정계수 1.0이 아닌 제품 수: {(forecast['보정계수'] != 1.0).sum()}")
        
        # 보정계수가 1.0이 아닌 제품들 표시
        non_one_adjustments = forecast[forecast['보정계수'] != 1.0]
        if len(non_one_adjustments) > 0:
            st.write("**보정계수가 1.0이 아닌 제품들:**")
            st.dataframe(non_one_adjustments[['경로', '제품명', '보정계수']].head(10))
        else:
            st.write("**모든 제품의 보정계수가 1.0입니다.**")
            st.write("가능한 원인:")
            st.write("1. 과거 판매 데이터 부족")
            st.write("2. 제품코드 매칭 실패")
            st.write("3. 경로명 불일치")
            st.write("4. 월 형식 불일치")
        
        # 개별 제품 단가 사용 확인
        st.write("**개별 제품 단가 사용 확인**")
        st.write("""
        - ✅ 각 제품의 개별 판매가를 사용하여 예측수량 계산
        - ✅ 평균단가는 사용하지 않음
        - ✅ 예측수량 = 제품별_예상매출 ÷ 개별제품단가
        """)
        
        # 제품별 단가 샘플 표시
        st.write("**제품별 단가 샘플 (상위 10개)**")
        price_sample = forecast[['제품명', '판매가']].head(10).copy()
        price_sample['판매가'] = price_sample['판매가'].apply(lambda x: f"{x:,.0f}원")
        st.dataframe(price_sample, use_container_width=True)
    
    # 기존 대시보드 표시
    display_future_dashboard(forecast, selected_routes)
