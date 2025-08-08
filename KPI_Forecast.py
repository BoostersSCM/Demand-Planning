import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows 기본 한글 폰트
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

# ========================
# 1. CSV 파일 읽기
# ========================

product_info = pd.read_csv('product_info.csv', encoding='utf-8')
sales_history = pd.read_csv('sales_history.csv', encoding='utf-8')
kpi_history = pd.read_csv('kpi_history.csv', encoding='utf-8')

# 판매가 컬럼을 숫자형으로 변환 (쉼표 제거)
product_info['판매가'] = product_info['판매가'].str.replace(',', '').astype(float)

# KPI매출 컬럼의 쉼표 제거 및 숫자형 변환
kpi_history['KPI매출'] = kpi_history['KPI매출'].str.replace(',', '').astype(float)

# 최신 KPI (예시: 2025-08)
kpi_current = pd.DataFrame({
    '월': ['25-Aug', '25-Aug'],
    '경로': ['Amazon(USA)', 'B2B(GLOBAL)'],
    'KPI매출': [910171022, 3000000000]
})

# ========================
# 2. 과거 보정계수 계산 함수
# ========================

def calculate_adjustment_factors(sales_df, kpi_df, product_df, target_month):
    sales = sales_df[sales_df['월'] == target_month]
    kpi = kpi_df[kpi_df['월'] == target_month]
    
    df = pd.merge(sales, product_df, on=['경로', '제품명'])
    df = pd.merge(df, kpi, on=['월', '경로'])
    
    df['매출'] = df['판매수량'] * df['판매가']
    
    df['경로_총수량'] = df.groupby(['경로'])['판매수량'].transform('sum')
    df['경로_총매출'] = df.groupby(['경로'])['매출'].transform('sum')
    
    df['경로_평균단가'] = df['경로_총매출'] / df['경로_총수량']
    df['경로_KPI수량'] = df['KPI매출'] / df['경로_평균단가']
    
    df['판매비중'] = df['판매수량'] / df['경로_총수량']
    df['예측수량'] = df['판매비중'] * df['경로_KPI수량']
    
    df['보정계수'] = df['판매수량'] / df['예측수량']
    
    return df[['경로', '제품명', '보정계수']]

adjustment_factors = calculate_adjustment_factors(sales_history, kpi_history, product_info, '25-Jul')

# ========================
# 3. 최신 KPI 기반 수요 예측 함수
# ========================

def calculate_dynamic_popularity_weights(df, sales_history, target_month):
    """
    과거 판매 데이터 기반으로 동적 인기도 가중치 계산 (경로별 구분)
    
    계산 방식:
    1. 과거 6개월 데이터에서 제품별 판매량 추이 분석
    2. 최근 2개월 vs 과거 6개월의 2개월 평균 비교로 추세 분석
    3. 경로별로 정규화하여 상대적 인기도 계산
    """
    # 과거 6개월 데이터로 판매량 추이 분석
    past_months = ['2025년 2월', '2025년 3월', '2025년 4월', '2025년 5월', '2025년 6월', '2025년 7월']
    
    # 인기도 가중치 초기화
    df['인기도_가중치'] = 1.0
    
    for route in df['경로'].unique():
        print(f"\n=== {route} 경로 인기도 가중치 계산 ===")
        
        # 해당 경로의 과거 실제 판매 데이터
        route_sales = sales_history[
            (sales_history['경로'] == route) &
            (sales_history['월'].isin(past_months))
        ]
        
        if len(route_sales) > 0:
            # 제품코드 기반으로 판매량 추이 분석
            if '제품코드' in route_sales.columns and '제품코드' in df.columns:
                valid_sales = route_sales[route_sales['제품코드'].notna() & (route_sales['제품코드'] != '')]
                
                if len(valid_sales) > 0:
                    # 제품별 월간 판매량 계산
                    monthly_sales = valid_sales.groupby(['제품코드', '월'])['판매수량'].sum().reset_index()
                    
                    # 제품별 판매량 계산 (최근 2개월 vs 과거 6개월의 2개월 평균)
                    product_total_sales = valid_sales.groupby('제품코드')['판매수량'].sum()
                    
                    # 최근 2개월 판매량 (6월, 7월)
                    recent_2months_sales = valid_sales[valid_sales['월'].isin(['2025년 6월', '2025년 7월'])].groupby('제품코드')['판매수량'].sum()
                    
                    # 과거 4개월 판매량 (2월, 3월, 4월, 5월)
                    past_4months_sales = valid_sales[valid_sales['월'].isin(['2025년 2월', '2025년 3월', '2025년 4월', '2025년 5월'])].groupby('제품코드')['판매수량'].sum()
                    
                    # 제품별 인기도 점수 계산
                    popularity_scores = {}
                    
                    for product_code in df[df['경로'] == route]['제품코드'].unique():
                        if pd.isna(product_code) or product_code == '':
                            popularity_scores[product_code] = 1.0
                            continue
                            
                        total_sales = product_total_sales.get(product_code, 0)
                        recent_2months = recent_2months_sales.get(product_code, 0)
                        past_4months = past_4months_sales.get(product_code, 0)
                        
                        # 판매량이 없는 제품은 기본값
                        if total_sales == 0:
                            popularity_scores[product_code] = 1.0
                            continue
                        
                        # 과거 4개월의 2개월 평균 판매량
                        past_2months_avg = past_4months / 2 if past_4months > 0 else 0
                        
                        # 최근 2개월 vs 과거 2개월 평균 비교 (추세 비율)
                        if past_2months_avg > 0:
                            trend_ratio = recent_2months / past_2months_avg
                        else:
                            trend_ratio = 1.0  # 과거 데이터가 없으면 중립
                        
                        # 판매량 규모 점수 (전체 대비 비중)
                        total_route_sales = product_total_sales.sum()
                        volume_score = total_sales / total_route_sales if total_route_sales > 0 else 0
                        
                        # 인기도 점수 = 판매량 규모 × 추세 비율
                        popularity_score = volume_score * trend_ratio
                        
                        # 최소값 보장 (0.5 이상)
                        popularity_score = max(0.5, popularity_score)
                        
                        popularity_scores[product_code] = popularity_score
                        
                        print(f"  {product_code}: 총판매량={total_sales:,}, 최근2개월={recent_2months:,}, "
                              f"과거4개월={past_4months:,}, 추세비율={trend_ratio:.2f}, 인기도점수={popularity_score:.3f}")
                    
                    # 인기도 점수를 가중치로 변환 (1.0 기준으로 정규화)
                    if popularity_scores:
                        max_score = max(popularity_scores.values())
                        min_score = min(popularity_scores.values())
                        
                        # 가중치 범위 조정 (0.7 ~ 1.3)
                        for product_code, score in popularity_scores.items():
                            if max_score > min_score:
                                # 정규화 후 범위 조정
                                normalized_score = (score - min_score) / (max_score - min_score)
                                weight = 0.7 + (normalized_score * 0.6)  # 0.7 ~ 1.3 범위
                            else:
                                weight = 1.0
                            
                            popularity_scores[product_code] = round(weight, 2)
                    
                    # 가중치 적용
                    for _, row in df[df['경로'] == route].iterrows():
                        product_code = row['제품코드']
                        weight = popularity_scores.get(product_code, 1.0)
                        df.loc[(df['경로'] == route) & (df['제품코드'] == product_code), '인기도_가중치'] = weight
                        
                        product_name = row['제품명']
                        print(f"  {product_name} ({product_code}): 가중치 {weight}")
                else:
                    # 판매 데이터가 없는 경우 기본값
                    print(f"  판매 데이터 없음 - 기본 가중치 1.0 적용")
            else:
                # 제품코드가 없는 경우 제품명 기반 계산
                product_total_sales = route_sales.groupby('제품명')['판매수량'].sum()
                
                # 최근 2개월 판매량 (6월, 7월)
                recent_2months_sales = route_sales[route_sales['월'].isin(['2025년 6월', '2025년 7월'])].groupby('제품명')['판매수량'].sum()
                
                # 과거 4개월 판매량 (2월, 3월, 4월, 5월)
                past_4months_sales = route_sales[route_sales['월'].isin(['2025년 2월', '2025년 3월', '2025년 4월', '2025년 5월'])].groupby('제품명')['판매수량'].sum()
                
                popularity_scores = {}
                total_route_sales = product_total_sales.sum()
                
                for product_name in df[df['경로'] == route]['제품명'].unique():
                    total_sales = product_total_sales.get(product_name, 0)
                    recent_2months = recent_2months_sales.get(product_name, 0)
                    past_4months = past_4months_sales.get(product_name, 0)
                    
                    if total_sales == 0:
                        popularity_scores[product_name] = 1.0
                        continue
                    
                    # 과거 4개월의 2개월 평균 판매량
                    past_2months_avg = past_4months / 2 if past_4months > 0 else 0
                    
                    # 최근 2개월 vs 과거 2개월 평균 비교 (추세 비율)
                    if past_2months_avg > 0:
                        trend_ratio = recent_2months / past_2months_avg
                    else:
                        trend_ratio = 1.0  # 과거 데이터가 없으면 중립
                    
                    volume_score = total_sales / total_route_sales if total_route_sales > 0 else 0
                    popularity_score = volume_score * trend_ratio
                    popularity_score = max(0.5, popularity_score)
                    
                    popularity_scores[product_name] = popularity_score
                
                # 가중치 정규화 및 적용
                if popularity_scores:
                    max_score = max(popularity_scores.values())
                    min_score = min(popularity_scores.values())
                    
                    for product_name, score in popularity_scores.items():
                        if max_score > min_score:
                            normalized_score = (score - min_score) / (max_score - min_score)
                            weight = 0.7 + (normalized_score * 0.6)
                        else:
                            weight = 1.0
                        
                        popularity_scores[product_name] = round(weight, 2)
                
                for _, row in df[df['경로'] == route].iterrows():
                    product_name = row['제품명']
                    weight = popularity_scores.get(product_name, 1.0)
                    df.loc[(df['경로'] == route) & (df['제품명'] == product_name), '인기도_가중치'] = weight
        else:
            print(f"  과거 데이터 없음 - 기본 가중치 1.0 적용")
    
    return df

def estimate_demand(kpi_df, product_df, adjustment_df, sales_history=None, target_month=None):
    df = pd.merge(product_df, kpi_df, on='경로')
    
    # 경로별 평균 판매가 계산 (안전한 방식)
    avg_price = product_df.groupby('경로')['판매가'].mean().to_dict()
    df['경로_평균단가'] = df['경로'].map(avg_price)
    
    # NaN 값 처리
    df['경로_평균단가'] = df['경로_평균단가'].fillna(df['경로_평균단가'].mean())
    
    df['경로_KPI수량'] = df['KPI매출'] / df['경로_평균단가']
    
    # 제품별 차별화된 비중 계산 (판매가 기반 가중 평균)
    # 높은 가격 제품일수록 수량은 적지만 매출 기여도는 높음
    df['경로_총가격'] = df.groupby('경로')['판매가'].transform('sum')
    df['판매비중'] = df['판매가'] / df['경로_총가격']
    
    # 동적 인기도 가중치 계산 (과거 판매 데이터 기반)
    # 과거 3개월 데이터로 판매량 추이 분석
    past_months = ['2025년 5월', '2025년 6월', '2025년 7월']
    
    # 인기도 가중치 초기화
    df['인기도_가중치'] = 1.0
    
    # 동적 인기도 가중치 계산 (경로별 구분)
    # 실제 판매 데이터가 없는 경우를 위해 기본값 설정
    try:
        # sales_history 데이터가 있는 경우 동적 계산
        if sales_history is not None:
            df = calculate_dynamic_popularity_weights(df, sales_history, target_month)
        else:
            # sales_history가 없는 경우 기본 가중치 사용
            popularity_weights = {
                '바쿠치올플럼핑세럼[30ml/-]': 1.2,
                '비타민일루미네이팅세럼[30ml/-]': 1.1,
                '스위밍풀토너[300ml/-]': 1.0,
                '스위밍풀앰플[50ml/-]': 0.9,
                '퍼플PDRN포어세럼[30ml/-]': 1.0,
                '퍼플라이스포어팩클렌저[130g/-]': 0.8,
                '퍼플라이스포어클렌징오일[200ml/-]': 0.8,
                '스위밍풀토너패드[220ml|70매/-]': 0.7,
                '콜라겐 하이드로겔 마스크[4EA|30g/타이트업]': 0.9,
                '콜라겐 하이드로겔 마스크[4EA|30g/글로우업]': 0.9,
                '이퀄베리-바쿠치올플럼핑캡슐크림[50ml/-]': 1.1,
                '스위밍풀토너[155ml/-]': 0.8
            }
            df['인기도_가중치'] = df['제품명'].map(lambda x: popularity_weights.get(x, 1.0))
    except Exception as e:
        print(f"동적 인기도 가중치 계산 중 오류 발생: {e}")
        # 오류 발생 시 기본 가중치 사용
        df['인기도_가중치'] = 1.0
    
    # 최종 판매 비중 = 기본 비중 × 인기도 가중치
    df['판매비중'] = df['판매비중'] * df['인기도_가중치']
    
    # 경로별로 정규화 (총합이 1이 되도록)
    df['판매비중'] = df['판매비중'] / df.groupby('경로')['판매비중'].transform('sum')
    
    df['예측수량'] = df['판매비중'] * df['경로_KPI수량']
    
    df = pd.merge(df, adjustment_df, on=['경로', '제품명'], how='left')
    df['보정계수'] = df['보정계수'].fillna(1.0)
    
    df['보정수량'] = df['예측수량'] * df['보정계수']
    
    return df[['월', '경로', '제품명', '판매가', '예측수량', '보정계수', '보정수량', '판매비중']]

forecast = estimate_demand(kpi_current, product_info, adjustment_factors, sales_history, '25-Aug')

# ========================
# 4. 결과 출력
# ========================

print("\n=== 보정계수 ===")
print(adjustment_factors)

print("\n=== 최신 KPI 기반 수요 예측 결과 ===")
print(forecast)

# 예측 월 정보 출력
print(f"\n📅 예측 대상 월: {forecast['월'].iloc[0]}")
print(f"📊 예측 경로: {', '.join(forecast['경로'].unique())}")

# 제품별 분포 출력
print(f"\n📦 제품별 예측 수량 분포:")
for route in forecast['경로'].unique():
    route_data = forecast[forecast['경로'] == route]
    print(f"\n🔸 {route}:")
    for _, row in route_data.iterrows():
        print(f"  • {row['제품명']}: {row['보정수량']:,.0f}개 (비중: {row['판매비중']:.1%})")

# ========================
# 5. 시각화
# ========================

# 그래프 크기 설정
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))

# 1. 경로별 총 예측 수량
route_totals = forecast.groupby('경로')['보정수량'].sum()
colors = ['#FF6B6B', '#4ECDC4']
bars = ax1.bar(route_totals.index, route_totals.values, color=colors)
ax1.set_title('경로별 총 예측 수량', fontsize=14, fontweight='bold')
ax1.set_ylabel('예측 수량', fontsize=12)
ax1.tick_params(axis='x', rotation=45)
for i, v in enumerate(route_totals.values):
    ax1.text(i, v + max(route_totals.values) * 0.01, f'{v:,.0f}', 
              ha='center', va='bottom', fontweight='bold')

# 2. 제품별 예측 수량 (상위 10개)
product_totals = forecast.groupby('제품명')['보정수량'].sum().sort_values(ascending=False).head(10)
bars2 = ax2.barh(range(len(product_totals)), product_totals.values, color='#45B7D1')
ax2.set_title('제품별 예측 수량 (상위 10개)', fontsize=14, fontweight='bold')
ax2.set_xlabel('예측 수량', fontsize=12)
ax2.set_yticks(range(len(product_totals)))
ax2.set_yticklabels([name[:20] + '...' if len(name) > 20 else name for name in product_totals.index])
for i, v in enumerate(product_totals.values):
    ax2.text(v + max(product_totals.values) * 0.01, i, f'{v:,.0f}', 
              ha='left', va='center', fontweight='bold')

# 3. 경로별 제품 수량 분포
pivot_data = forecast.pivot_table(index='경로', columns='제품명', values='보정수량', aggfunc='sum')
pivot_data = pivot_data.fillna(0)
im = ax3.imshow(pivot_data.values, cmap='YlOrRd', aspect='auto')
ax3.set_title('경로별 제품 수량 분포 히트맵', fontsize=14, fontweight='bold')
ax3.set_xlabel('제품', fontsize=12)
ax3.set_ylabel('경로', fontsize=12)
ax3.set_xticks(range(len(pivot_data.columns)))
ax3.set_xticklabels([col[:15] + '...' if len(col) > 15 else col for col in pivot_data.columns], rotation=45)
ax3.set_yticks(range(len(pivot_data.index)))
ax3.set_yticklabels(pivot_data.index)
plt.colorbar(im, ax=ax3, label='예측 수량')

# 4. 경로별 평균 단가와 예측 수량 관계
route_avg_price = forecast.groupby('경로')['판매가'].mean()
route_avg_quantity = forecast.groupby('경로')['보정수량'].mean()
scatter = ax4.scatter(route_avg_price, route_avg_quantity, s=200, alpha=0.7, 
                      c=colors)
ax4.set_title('경로별 평균 단가 vs 예측 수량', fontsize=14, fontweight='bold')
ax4.set_xlabel('평균 단가 (원)', fontsize=12)
ax4.set_ylabel('평균 예측 수량', fontsize=12)
for i, route in enumerate(route_avg_price.index):
    ax4.annotate(route, (route_avg_price.iloc[i], route_avg_quantity.iloc[i]),
                 xytext=(5, 5), textcoords='offset points', fontweight='bold')

plt.tight_layout()
plt.show()

# ========================
# 6. 요약 통계 출력
# ========================

print("\n" + "="*60)
print("📊 KPI 기반 수요 예측 요약 통계")
print("="*60)

print(f"\n📈 경로별 예측 요약:")
for route in forecast['경로'].unique():
    route_data = forecast[forecast['경로'] == route]
    total_quantity = route_data['보정수량'].sum()
    avg_price = route_data['판매가'].mean()
    total_revenue = (route_data['보정수량'] * route_data['판매가']).sum()
    print(f"• {route}:")
    print(f"  - 총 예측 수량: {total_quantity:,.0f}개")
    print(f"  - 평균 단가: {avg_price:,.0f}원")
    print(f"  - 예상 매출: {total_revenue:,.0f}원")

print(f"\n📋 전체 요약:")
print(f"• 총 예측 수량: {forecast['보정수량'].sum():,.0f}개")
print(f"• 평균 예측 수량: {forecast['보정수량'].mean():,.0f}개")
print(f"• 최대 예측 수량: {forecast['보정수량'].max():,.0f}개")
print(f"• 최소 예측 수량: {forecast['보정수량'].min():,.0f}개")

print("\n" + "="*60)
