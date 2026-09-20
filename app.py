"""Airline Customer Experience & Commercial Insights Copilot (Streamlit app).

Reads precomputed files from ./data (built in the notebooks). Nothing heavy is computed at startup.
"""
import os
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title='Airline CX Copilot', page_icon='✈️', layout='wide')

DATA = Path(__file__).parent / 'data'
ASPECTS = ['seat', 'crew', 'food', 'ground_baggage', 'delays', 'value']
CABINS = ['Economy Class', 'Premium Economy', 'Business Class', 'First Class']
EVAL_FILE = 'llm_eval_openai_gpt-oss-20b.csv'


# ----------------------------------------------------------------------------- helpers
def need(*names):
    """Return True if all files exist, else show which are missing."""
    missing = [n for n in names if not (DATA / n).exists()]
    if missing:
        st.warning('This tab needs files that are missing from the repo `data/` folder: ' + ', '.join(missing))
        return False
    return True


@st.cache_data
def load_parquet(name):
    return pd.read_parquet(DATA / name)


@st.cache_data
def load_csv(name, **kwargs):
    return pd.read_csv(DATA / name, **kwargs)


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


def error_bar_chart(df, y, x, lo, hi, x_title):
    ax = alt.Axis(labelOverlap=False, labelLimit=320, title=None)
    base = alt.Chart(df)
    rule = base.mark_rule().encode(y=alt.Y(f'{y}:N', sort=None, axis=ax), x=alt.X(f'{lo}:Q', title=x_title), x2=f'{hi}:Q')
    pts = base.mark_point(filled=True, size=70).encode(
        y=alt.Y(f'{y}:N', sort=None, axis=ax), x=f'{x}:Q',
        tooltip=[y, alt.Tooltip(x, format='.3f'), alt.Tooltip(lo, format='.3f'), alt.Tooltip(hi, format='.3f')])
    # total chart height includes the axis area, so add room on top of the rows
    return (rule + pts).properties(width='container', height=30 * len(df) + 100)


# ----------------------------------------------------------------------------- tab 1: insights
def tab_insights():
    if not need('topic_summary.csv', 'aspect_summary.csv', 'reviews_tagged.parquet', EVAL_FILE):
        return
    st.subheader('What passengers talk about')
    ts = load_csv('topic_summary.csv')
    chart = alt.Chart(ts).mark_bar().encode(
        x=alt.X('avg_rating:Q', title='Average overall rating (1-10)'),
        y=alt.Y('topic_name:N', sort='x', title=None, axis=alt.Axis(labelLimit=340, labelOverlap=False)),
        tooltip=['topic_name', 'n_reviews', 'avg_rating', 'recommend_rate', 'top_terms']).properties(width='container', height=440)
    st.altair_chart(chart)
    st.caption('Topics come from MiniLM sentence embeddings clustered with k-means (k = 12). Clusters overlap (silhouette about 0.03), '
               'so read them as useful groupings rather than sharp categories. Two clusters are driven by place names, not complaint types.')

    st.subheader('Aspect-level sentiment (LLM-tagged)')
    tagged = load_parquet('reviews_tagged.parquet')
    summ = pd.DataFrame({
        'mentioned in % of reviews': [(tagged[a] != 'not_mentioned').mean() * 100 for a in ASPECTS],
        '% of mentions negative': [(tagged.loc[tagged[a] != 'not_mentioned', a] == 'negative').mean() * 100 for a in ASPECTS],
    }, index=ASPECTS).round(1)
    c1, c2 = st.columns(2)
    c1.dataframe(summ)
    c2.bar_chart(summ['% of mentions negative'])
    st.caption(f'Based on {len(tagged)} tagged reviews. Reviewers skew negative (65% did not recommend), so negativity is high overall.')

    st.markdown('**Negative mentions by cabin** (share of that cabin\'s reviews with a negative mention)')
    by_cabin = pd.DataFrame({a: (tagged[a] == 'negative').groupby(tagged['seat_type']).mean() * 100 for a in ASPECTS}).round(1)
    by_cabin.insert(0, 'n reviews', tagged.groupby('seat_type').size())
    st.dataframe(by_cabin)
    st.caption('Cabins with few reviews (First Class, Premium Economy) are not reliable. Compare Economy and Business only, and treat it as suggestive.')

    st.subheader('How reliable is the tagging?')
    ev = load_csv(EVAL_FILE)
    WEAK = {'seat': 'seat_comfort', 'crew': 'cabin_staff_service', 'food': 'food_beverages',
            'ground_baggage': 'ground_service', 'value': 'value_for_money'}
    rows = []
    for a, col in WEAK.items():
        if col not in tagged.columns:
            continue
        sub = tagged[tagged[a].isin(['positive', 'negative']) & tagged[col].notna() & (tagged[col] != 3)]
        agree = ((sub[a] == 'positive') & (sub[col] >= 4)) | ((sub[a] == 'negative') & (sub[col] <= 2))
        rows.append({'aspect': a, 'n': len(sub), 'agreement with passenger sub-rating': round(agree.mean(), 3)})
    e1, e2 = st.columns(2)
    e1.markdown('**Against a 100-review gold set** (labels drafted by a larger model; disputed labels re-decided by me, so scores are optimistic)')
    e1.dataframe(ev)
    e2.markdown('**Independent check:** model polarity vs the passenger\'s own sub-rating')
    e2.dataframe(pd.DataFrame(rows))


# ----------------------------------------------------------------------------- tab 2: drivers
def tab_drivers():
    if not need('odds_ratios.csv', 'airline_stats.csv', 'stat_tests_summary.csv'):
        return
    src = 'reviews_clean.parquet' if (DATA / 'reviews_clean.parquet').exists() else 'reviews_topics.parquet'
    if not need(src):
        return
    df = load_parquet(src)

    st.subheader('Recommend rate by cabin')
    g = df.dropna(subset=['seat_type']).groupby('seat_type')['recommended_bin'].agg(['sum', 'count']).reset_index()
    g['rate'] = g['sum'] / g['count']
    g[['ci_low', 'ci_high']] = [wilson(k, n) for k, n in zip(g['sum'], g['count'])]
    st.altair_chart(error_bar_chart(g.sort_values('rate'), 'seat_type', 'rate', 'ci_low', 'ci_high', 'Recommend rate (95% CI)'))
    st.caption(f'Computed from {len(df):,} reviews. Business/First cabins are recommended about 22 points more than Economy; Premium Economy is not different from Economy.')

    st.subheader('What drives "would recommend"?')
    orr = load_csv('odds_ratios.csv', index_col=0)
    feat = [f for f in ['seat_comfort', 'cabin_staff_service', 'food_beverages', 'ground_service', 'value_for_money'] if f in orr.index]
    fo = orr.loc[feat].reset_index().rename(columns={'index': 'factor'}).sort_values('odds_ratio')
    fo['factor'] = fo['factor'].astype(str)
    fchart = error_bar_chart(fo, 'factor', 'odds_ratio', 'ci_low', 'ci_high', 'Odds ratio per +1 star (95% CI)')
    st.altair_chart(fchart)
    st.caption('Logistic regression on sub-ratings, adjusted for cabin and traveller type (complete cases only). '
               'These are associations, not causes. Value for money is the strongest.')

    st.subheader('Airlines')
    a = load_csv('airline_stats.csv')
    a = a.sort_values('rate')
    top = pd.concat([a.head(10), a.tail(10)])
    st.altair_chart(error_bar_chart(top, 'airline_name', 'rate', 'ci_low', 'ci_high', 'Recommend rate (95% CI)'))
    n_sig = int(a['differs_from_average'].sum()) if 'differs_from_average' in a.columns else 0
    st.caption(f'{n_sig} of {len(a)} airlines differ from the overall rate after multiple-testing correction. '
               'Each airline has about 100 reviews (interval about +/-0.09), so rank tiers, not exact ranks.')
    with st.expander('All airlines'):
        st.dataframe(a.round(3))

    st.subheader('Trend over time')
    yr = pd.to_datetime(df['date_flown'], errors='coerce').dt.year
    t = df.assign(year=yr).dropna(subset=['year']).groupby('year')['recommended_bin'].agg(['sum', 'count'])
    t = t[t['count'] >= 100].copy()
    if len(t):
        t['rate'] = t['sum'] / t['count']
        t.index = t.index.astype(int).astype(str)
        st.line_chart(t['rate'])
    st.caption('The pooled decline is mostly an airline-mix effect. Among 42 airlines with enough reviews in both periods, the average change was -4.8 points '
               '(60% declined; Wilcoxon p = 0.048, borderline).')

    st.subheader('Statistical tests')
    tests = load_csv('stat_tests_summary.csv').copy()
    tests['p_value'] = tests['p_value'].map(lambda p: f'{p:.2e}')
    st.dataframe(tests)


# ----------------------------------------------------------------------------- tab 3: optimizer
DEFAULT_P = {'Refunds, cancellations & customer service': 0.50, 'Severe service failures & long delays': 0.25,
             'Lost & delayed baggage': 0.60, 'Flight delays & missed connections': 0.35,
             'Baggage fees & carry-on charges': 0.40, 'Seating, boarding & staff handling': 0.30}
DEFAULT_H = {'Refunds, cancellations & customer service': 1.0, 'Severe service failures & long delays': 1.5,
             'Lost & delayed baggage': 1.5, 'Flight delays & missed connections': 0.5,
             'Baggage fees & carry-on charges': 0.5, 'Seating, boarding & staff handling': 0.5}


def build_segments(topics, value_index, assump):
    atrisk = topics[topics['topic_name'].isin(assump['topic_name']) & (topics['recommended_bin'] == 0) & topics['seat_type'].isin(value_index)]
    seg = atrisk.groupby(['topic_name', 'seat_type']).size().rename('cases').reset_index()
    seg['value'] = seg['seat_type'].map(value_index)
    seg = seg.merge(assump, on='topic_name', how='left')
    seg['gain_per_case'] = seg['value'] * seg['recovery_p']
    seg['gain_per_hour'] = seg['gain_per_case'] / seg['hours']
    seg['segment'] = seg['topic_name'] + ' | ' + seg['seat_type']
    return seg.reset_index(drop=True)


def solve_plan(seg, capacity, min_cover=0.0):
    import pulp
    prob = pulp.LpProblem('recovery', pulp.LpMaximize)
    x = [pulp.LpVariable(f'x{i}', lowBound=0, upBound=int(c), cat='Integer') for i, c in enumerate(seg['cases'])]
    prob += pulp.lpSum(float(g) * xi for g, xi in zip(seg['gain_per_case'], x))
    prob += pulp.lpSum(float(h) * xi for h, xi in zip(seg['hours'], x)) <= capacity
    if min_cover > 0:
        for xi, c in zip(x, seg['cases']):
            prob += xi >= int(np.floor(min_cover * c))
    status = prob.solve(pulp.PULP_CBC_CMD(msg=0))
    if pulp.LpStatus[status] != 'Optimal':
        return None
    return np.array([xi.value() for xi in x])


def plan_proportional(seg, capacity):
    frac = min(1.0, capacity / float((seg['cases'] * seg['hours']).sum()))
    return seg['cases'].values * frac


def plan_premium_first(seg, capacity):
    rank = seg['seat_type'].map({'First Class': 0, 'Business Class': 1, 'Premium Economy': 2, 'Economy Class': 3})
    order = seg.assign(rank=rank).sort_values(['rank', 'hours'])
    plan = np.zeros(len(seg))
    left = capacity
    for i in order.index:
        n = min(float(seg.loc[i, 'cases']), left / float(seg.loc[i, 'hours']))
        plan[i] = max(n, 0.0)
        left -= plan[i] * float(seg.loc[i, 'hours'])
        if left <= 1e-9:
            break
    return plan


def value_of(seg, plan):
    return float((seg['gain_per_case'] * plan).sum())


@st.cache_data
def run_plans(seg, capacity_share, min_cover):
    total = float((seg['cases'] * seg['hours']).sum())
    cap = capacity_share * total
    plans = {'Proportional (first-come-first-served)': plan_proportional(seg, cap),
             'Premium-first rule': plan_premium_first(seg, cap),
             'Optimised (LP)': solve_plan(seg, cap, 0.0)}
    fair = solve_plan(seg, cap, min_cover) if min_cover > 0 else None
    if fair is not None:
        plans[f'Optimised (LP, min {min_cover:.0%} coverage)'] = fair
    return {k: v for k, v in plans.items() if v is not None}, total, cap


@st.cache_data
def capacity_curve(seg, min_cover):
    total = float((seg['cases'] * seg['hours']).sum())
    rows = []
    for sh in [0.10, 0.15, 0.25, 0.35, 0.50, 0.75, 1.0]:
        cap = sh * total
        opt = solve_plan(seg, cap, 0.0)
        rows.append({'capacity share': sh, 'proportional': value_of(seg, plan_proportional(seg, cap)),
                     'premium-first': value_of(seg, plan_premium_first(seg, cap)),
                     'optimised': value_of(seg, opt) if opt is not None else np.nan})
    return pd.DataFrame(rows).set_index('capacity share')


def tab_optimizer():
    if not need('reviews_topics.parquet'):
        return
    topics = load_parquet('reviews_topics.parquet')
    st.subheader('Service-recovery capacity planner')
    st.warning('Case counts come from the review data. **Revenue values, recovery rates and handling times below are illustrative assumptions.** '
               'Change them to your own numbers and the plan updates.')

    rate = topics.groupby('topic_name')['recommended_bin'].mean()
    complaint = rate[rate < 0.30].index.tolist()
    if not complaint:
        st.info('No complaint topics found in the data.')
        return

    c1, c2 = st.columns([1, 2])
    with c1:
        cap_share = st.slider('Team capacity (% of hours needed for all at-risk cases)', 5, 100, 25, step=5) / 100
        min_cover = st.slider('Fairness floor: minimum share of each segment handled', 0, 30, 10, step=5) / 100
        st.markdown('**Relative revenue per passenger** (Economy = 1)')
        vi = {'Economy Class': st.number_input('Economy', 0.1, 50.0, 1.0, step=0.1),
              'Premium Economy': st.number_input('Premium Economy', 0.1, 50.0, 1.6, step=0.1),
              'Business Class': st.number_input('Business', 0.1, 50.0, 4.0, step=0.1),
              'First Class': st.number_input('First', 0.1, 50.0, 8.0, step=0.1)}
    with c2:
        st.markdown('**Recovery probability and hours per case, by complaint topic** (editable)')
        base = pd.DataFrame({'topic_name': complaint,
                             'recovery_p': [DEFAULT_P.get(t, 0.40) for t in complaint],
                             'hours': [DEFAULT_H.get(t, 1.0) for t in complaint]})
        assump = st.data_editor(base, disabled=['topic_name'], key='assump_editor')
    assump = assump.dropna()
    assump = assump[(assump['recovery_p'] > 0) & (assump['hours'] > 0)]

    seg = build_segments(topics, vi, assump)
    if seg.empty:
        st.info('No segments to plan with these settings.')
        return
    plans, total, cap = run_plans(seg, cap_share, min_cover)

    st.markdown(f'**{len(seg)} segments, {int(seg["cases"].sum()):,} at-risk cases.** Hours needed for all: {total:,.0f}. Capacity: {cap:,.0f}.')
    summary = pd.DataFrame({'expected retained value': {k: value_of(seg, v) for k, v in plans.items()},
                            'hours used': {k: float((seg['hours'] * v).sum()) for k, v in plans.items()},
                            'cases handled': {k: float(v.sum()) for k, v in plans.items()}}).round(1)
    base_val = summary.loc['Proportional (first-come-first-served)', 'expected retained value']
    summary['vs proportional'] = ((summary['expected retained value'] / base_val - 1) * 100).round(1).astype(str) + '%'
    st.dataframe(summary)
    st.bar_chart(summary['expected retained value'])

    key = [k for k in plans if k.startswith('Optimised')][-1]
    alloc = seg.assign(handled=plans[key].round(1))
    alloc['share handled'] = (alloc['handled'] / alloc['cases']).round(2)
    st.markdown(f'**Allocation ({key})**')
    st.dataframe(alloc.sort_values('gain_per_hour', ascending=False)[['segment', 'cases', 'gain_per_hour', 'handled', 'share handled']].round(2))
    st.markdown('**Value vs team capacity**')
    st.line_chart(capacity_curve(seg, min_cover))
    st.caption('A simple premium-first rule usually captures most of the optimum when cabin values differ a lot; '
               'the LP adds most when capacity is scarce and handling times differ.')


# ----------------------------------------------------------------------------- tab 4: copilot
@st.cache_resource
def load_index():
    import faiss
    return faiss.read_index(str(DATA / 'faiss.index'))


@st.cache_resource
def load_embedder():
    try:
        from fastembed import TextEmbedding
        return TextEmbedding(model_name='sentence-transformers/all-MiniLM-L6-v2')
    except Exception:
        return None


@st.cache_resource
def load_tfidf():
    from sklearn.feature_extraction.text import TfidfVectorizer
    texts = load_parquet('reviews_topics.parquet')['review_text'].tolist()
    vec = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), sublinear_tf=True, max_features=60000)
    return vec, vec.fit_transform(texts)


def retrieve(query, k, cabins, rating_range):
    df = load_parquet('reviews_topics.parquet')
    embedder = load_embedder() if (DATA / 'faiss.index').exists() else None
    if embedder is not None:
        qv = np.array(list(embedder.embed([query])), dtype='float32')
        qv /= np.linalg.norm(qv, axis=1, keepdims=True)
        scores, ids = load_index().search(qv, min(400, len(df)))
        ids, scores = ids[0], scores[0]
        mode = 'semantic search (MiniLM embeddings + FAISS)'
    else:
        vec, X = load_tfidf()
        sims = (X @ vec.transform([query]).T).toarray().ravel()
        ids = np.argsort(-sims)[:400]
        scores = sims[ids]
        mode = 'keyword fallback (TF-IDF); the embedding model could not be loaded'
    res = df.iloc[ids].copy()
    res['score'] = scores
    res = res[res['seat_type'].isin(cabins) & res['overall_rating'].between(*rating_range)]
    return res.head(k), mode


def groq_key():
    try:
        key = st.secrets.get('GROQ_API_KEY')
    except Exception:
        key = None
    return key or os.environ.get('GROQ_API_KEY')


def answer_with_llm(question, hits, model):
    key = groq_key()
    if not key:
        return None, 'No GROQ_API_KEY found in secrets, so only the retrieved reviews are shown.'
    from groq import Groq
    context = '\n\n'.join(f'[{i + 1}] ({r.airline_name}, {r.seat_type}, rating {r.overall_rating:g}) {r.review_text[:700]}'
                          for i, r in enumerate(hits.itertuples()))
    system = ('You are an airline customer-experience analyst. Answer the question using ONLY the numbered passenger reviews provided. '
              'Cite the reviews you use like [1] or [2][3]. If the reviews do not contain enough to answer, say so plainly. '
              'Do not invent facts or numbers. Keep the answer under 150 words.')
    try:
        r = Groq(api_key=key).chat.completions.create(
            model=model, temperature=0, max_tokens=900,
            messages=[{'role': 'system', 'content': system},
                      {'role': 'user', 'content': f'Question: {question}\n\nReviews:\n{context}'}],
            extra_body={'reasoning_effort': 'low'})
        return r.choices[0].message.content, None
    except Exception as e:
        return None, f'The language model call failed ({str(e)[:160]}). The retrieved reviews are shown below.'


def tab_copilot():
    if not need('reviews_topics.parquet'):
        return
    st.subheader('Ask the reviews')
    st.caption('Retrieves the most relevant passenger reviews for your question, then a language model answers using only those reviews, with citations.')
    q = st.text_input('Question', placeholder='e.g. What do Economy passengers complain about most on baggage handling?', key='copilot_q')
    f1, f2, f3 = st.columns(3)
    cabins = f1.multiselect('Cabin', CABINS, default=CABINS)
    rating_range = f2.slider('Overall rating range', 1, 10, (1, 10))
    model = f3.selectbox('Model', ['openai/gpt-oss-20b', 'openai/gpt-oss-120b'])
    if st.button('Search', key='copilot_go') and q.strip():
        with st.spinner('Searching reviews...'):
            hits, mode = retrieve(q.strip(), 8, cabins, rating_range)
        st.caption(f'Retrieval: {mode}')
        if hits.empty:
            st.info('No reviews matched these filters.')
            return
        with st.spinner('Writing the answer...'):
            ans, err = answer_with_llm(q.strip(), hits, model)
        if ans:
            st.markdown('**Answer**')
            st.write(ans)
        if err:
            st.info(err)
        st.markdown('**Sources**')
        for i, r in enumerate(hits.itertuples(), 1):
            with st.expander(f'[{i}] {r.airline_name} | {r.seat_type} | rating {r.overall_rating:g} | topic: {r.topic_name} | similarity {r.score:.2f}'):
                st.write(r.review_text)


# ----------------------------------------------------------------------------- layout
st.title('✈️ Airline Customer Experience & Commercial Insights Copilot')
st.caption('Passenger reviews turned into topics, aspect sentiment, statistics and a recovery-capacity plan. '
           'Source: airlinequality.com reviews (Kaggle: juhibhojani/airline-reviews). Reviewers are self-selected and about 100 reviews per airline are included.')

t1, t2, t3, t4 = st.tabs(['Insights', 'Drivers', 'Optimizer', 'Copilot'])
with t1:
    tab_insights()
with t2:
    tab_drivers()
with t3:
    tab_optimizer()
with t4:
    tab_copilot()
