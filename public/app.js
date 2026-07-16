function switchTab(tabId) {
    const contents = document.querySelectorAll('.tab-content');
    contents.forEach(content => content.classList.add('hidden'));

    const buttons = document.querySelectorAll('.tab-btn');
    buttons.forEach(button => button.classList.remove('active'));

    document.getElementById(`${tabId}-card`).classList.remove('hidden');
    
    const targetButtonIndex = tabId === 'explanation' ? 0 : tabId === 'modules' ? 1 : 2;
    buttons[targetButtonIndex].classList.add('active');
}

document.getElementById('generate-btn').addEventListener('click', async () => {
    const topic = document.getElementById('topic').value.trim();
    const level = document.getElementById('level').value;

    const loader = document.getElementById('loader');
    const outputContainer = document.getElementById('output-container');

    const expContent = document.getElementById('explanation-content');
    const modContent = document.getElementById('modules-content');
    const resContent = document.getElementById('resources-content');

    if (!topic) {
        alert("Please enter a topic or course name.");
        return;
    }

    loader.classList.remove('hidden');
    outputContainer.classList.add('hidden');

    try {
        const response = await fetch('/api/generate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ topic, level })
        });

        const data = await response.json();

        if (response.ok) {
            expContent.innerHTML = marked.parse(data.explanation || 'No content provided.');
            modContent.innerHTML = marked.parse(data.modules || 'No content provided.');
            resContent.innerHTML = marked.parse(data.resources || 'No content provided.');

            outputContainer.classList.remove('hidden');
            switchTab('explanation');
        } else {
            alert(`Error: ${data.detail || 'An error occurred during generation.'}`);
        }
    } catch (error) {
        console.error("Communication error with API:", error);
        alert("Could not connect to backend server.");
    } finally {
        loader.classList.add('hidden');
    }
});