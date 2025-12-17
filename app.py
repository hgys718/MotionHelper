import streamlit as st
import pandas as pd
from transformers import pipeline
from PIL import Image

st.set_page_config(page_title="Motion Helper", page_icon="🎬", layout="centered")
st.title("🎬 Motion Helper")
st.write(
    "Up your motion graphics game with a little help from the AI. "
    "Generate motion graphic directions based on your project description or upload an image of a scene to get suggestions on which elements could be animated for motion graphics."

)
st.markdown("— *Model: `distilgpt2` from HuggingFace*")
@st.cache_data
def load_examples():
  df = pd.read_csv("prompt_examples.csv")
  return df
examples_df = load_examples()
with st.expander("📂 View example prompt dataset"):
    st.dataframe(examples_df)
@st.cache_resource
def load_generator():
    generator = pipeline(
        "text-generation",
        model="distilgpt2",
    )
    return generator
generator = load_generator()
@st.cache_resource
def load_vision_model():
    vision_pipeline = pipeline(
        "image-to-text",
        model="Salesforce/blip-image-captioning-base",
    )
    return vision_pipeline
vision_model = load_vision_model()
st.subheader("Describe your project")
project_title = st.text_input(
    "Project title",
    placeholder="e.g. Worn Beauty — title sequence about time and wrinkles"
)
project_description = st.text_area(
    "What is this motion piece about?",
    placeholder="Write 1–3 sentences about the mood, story, and visuals you want."

)
mood = st.selectbox(
    "Pick a main mood",
    ["worn", "calm", "energetic", "retro", "nostalgic", "playful", "futuristic"]
)
style_keywords = st.text_input(
    "Optional: keywords (comma-separated)",
    placeholder="e.g. grain, slow zoom, kinetic type"
)
st.caption("Tip: longer descriptions usually give more interesting prompts.")
generate_button = st.button("✨ Generate motion idea")
st.divider()
st.subheader("🖼️ Image Motion Analysis")
st.write("Upload an image of a scene to get suggestions on which elements could be animated for motion graphics.")
uploaded_file = st.file_uploader(
    "Choose an image file",
    type=["jpg", "jpeg", "png", "webp"],
    help="Upload a JPG, PNG, or WebP image to analyze"

)
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    analyze_button = st.button("🔍 Analyze Image for Motion Elements")
    if analyze_button:
        with st.spinner("Analyzing image and generating motion suggestions..."):
            image_description = vision_model(image)[0]["generated_text"]
            motion_prompt = (
                f"Image description: {image_description}\n\n"
                f"Based on this image, suggest specific elements that could be animated "
                f"or moved to create effective motion graphics. Focus on:\n"
                f"- Objects or elements that could move\n"
                f"- Camera movements that would work\n"
                f"- Transitions or effects\n"
                f"- Typography placement opportunities\n"
                f"- Textural or visual elements to animate\n\n"
                f"Provide 3-5 specific, actionable suggestions:\n"
            )
            motion_output = generator(
                motion_prompt,
                max_new_tokens=100,
                num_return_sequences=1,
                do_sample=True,
                top_p=0.9,
                temperature=0.8,
            )
            motion_suggestions = motion_output[0]["generated_text"]
            motion_text = motion_suggestions[len(motion_prompt):].strip()
        st.subheader("📸 Image Analysis")
        st.write(f"**Image Description:** {image_description}")
        st.subheader("🎬 Motion Suggestions")
        st.write(motion_text)
        st.subheader("💡 Quick Motion Tips")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Common Animated Elements:**")
            st.markdown("""
            - **Foreground objects**: Move independently
            - **Background elements**: Parallax scrolling
            - **Text layers**: Kinetic typography
            - **Shapes/forms**: Morphing or rotation
            """)
        with col2:
            st.markdown("**Camera Techniques:**")
            st.markdown("""
            - **Zoom in/out**: Focus on details
            - **Pan/tilt**: Reveal composition
            - **Depth of field**: Create focus shifts
            - **Tracking**: Follow moving elements
            """)
        st.info("💡 **Tip**: Use the image description above to inform your project description in the section above for more targeted motion suggestions!")
def get_related_examples(df, mood, max_rows=3):
    matches = df[df["mood"].str.contains(mood, case=False, na=False)]
    if matches.empty:
        return df.sample(min(max_rows, len(df)))
    return matches.sample(min(max_rows, len(matches)))
if generate_button:
    if not project_description.strip():
        st.error("Please write a project description first.")
    else:
        with st.spinner("Talking to the HuggingFace model..."):
            base_text = (
                f"Motion graphics concept.\n"
                f"Title: {project_title if project_title else 'Untitled Project'}\n"
                f"Mood: {mood}\n"
                f"Keywords: {style_keywords if style_keywords else 'none'}\n"
                f"Description: {project_description}\n\n"
                f"Describe a detailed motion direction with camera moves, "
                f"transitions, texture, and typography style in 3–5 sentences:\n"
            )
            out = generator(
                base_text,
                max_new_tokens=80,
                num_return_sequences=1,
                do_sample=True,
                top_p=0.9,
                temperature=0.9,
            )
            raw_text = out[0]["generated_text"]
        generated_part = raw_text[len(base_text):].strip()
        st.subheader("💡 Suggested motion direction")
        st.write(generated_part)
        st.subheader("📂 Related example prompts from dataset")
        related = get_related_examples(examples_df, mood)
        for idx, row in related.iterrows():
            st.markdown(f"**{row['project_title']}**  — *mood:* {row['mood']}")
            st.write(row["example_prompt"])
            st.markdown("---")