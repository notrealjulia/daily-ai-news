"""AI News dashboard. Read-only: it only displays what the pipeline has stored.

Run from the project root:  streamlit run app.py

This file is a thin renderer. There is no SQL, no pipeline code and no OpenAI code here;
everything it shows is worked out by ainews.dashboard, which reads the database through
a read-only connection. It has no controls that trigger anything.

Typography (base font size, heading sizes) is set in .streamlit/config.toml, not in CSS.
"""

import streamlit as st

from ainews import dashboard

st.set_page_config(page_title="AI News", layout="wide", initial_sidebar_state="collapsed")

data = dashboard.open_dashboard()

# Native Streamlit has no page-padding setting, so the whole page sits in the middle
# column of a 1 : 14 : 1 row; the outer two are just extra margin on either side.
_, page, _ = st.columns([1, 14, 1])

with page, st.container(gap="xsmall"):
    with st.container(horizontal=True, vertical_alignment="bottom"):
        st.title("AI News", width="content")
        if data is not None:
            st.caption(dashboard.format_header(data))

    if data is None:
        st.info(
            "No completed run yet. Run `python -m ainews cluster`, then `python -m ainews digest`, "
            "and reload this page."
        )
        st.stop()

    for row in dashboard.CATEGORY_GRID:
        for column, name in zip(st.columns(2, gap="small"), row):
            category = data.categories[name]
            # height="stretch" makes both cards in a row as tall as the taller one; the
            # stretch space then pushes the expander to the bottom of the card.
            with column.container(border=True, gap="xsmall", height="stretch"):
                st.subheader(dashboard.category_heading(category, data.total_stories))
                if not category.stories:
                    st.caption(dashboard.empty_text(data.window_hours))
                    continue
                if category.digest:
                    st.markdown(dashboard.escape_markdown(category.digest))
                st.space("stretch")
                with st.expander(dashboard.expander_label(category.story_count)):
                    for story in category.stories:
                        st.markdown(dashboard.story_markdown(story))
