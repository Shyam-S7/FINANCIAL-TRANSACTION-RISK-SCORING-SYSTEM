document.addEventListener('DOMContentLoaded', () => {
    // Generate particles
    const particlesContainer = document.getElementById('particles');
    for (let i = 0; i < 50; i++) {
        const particle = document.createElement('div');
        particle.style.position = 'absolute';
        particle.style.width = Math.random() * 3 + 'px';
        particle.style.height = particle.style.width;
        particle.style.background = Math.random() > 0.5 ? 'var(--neon-blue)' : 'var(--neon-red)';
        particle.style.left = Math.random() * 100 + 'vw';
        particle.style.top = Math.random() * 100 + 'vh';
        particle.style.opacity = Math.random() * 0.5 + 0.1;
        particle.style.boxShadow = `0 0 5px ${particle.style.background}`;
        particle.style.borderRadius = '50%';
        
        // Simple float animation
        const duration = Math.random() * 10 + 5;
        const delay = Math.random() * 5;
        particle.animate([
            { transform: 'translateY(0)' },
            { transform: 'translateY(-100px)' }
        ], {
            duration: duration * 1000,
            delay: delay * 1000,
            iterations: Infinity,
            direction: 'alternate',
            easing: 'ease-in-out'
        });
        
        particlesContainer.appendChild(particle);
    }

    const form = document.getElementById('fraudForm');
    const statusDisplay = document.getElementById('statusDisplay');
    const resultDisplay = document.getElementById('resultDisplay');
    const loadingState = document.getElementById('loadingState');
    const errorState = document.getElementById('errorState');
    const errorText = document.getElementById('errorText');
    
    const riskLabel = document.getElementById('riskLabel');
    const fraudProb = document.getElementById('fraudProb');
    const probFill = document.getElementById('probFill');

    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        // UI Transitions
        statusDisplay.classList.add('hidden');
        resultDisplay.classList.add('hidden');
        errorState.classList.add('hidden');
        loadingState.classList.remove('hidden');

        const formData = {
            type: document.getElementById('type').value,
            amount: parseFloat(document.getElementById('amount').value),
            oldbalanceOrg: parseFloat(document.getElementById('oldbalanceOrg').value),
            newbalanceOrig: parseFloat(document.getElementById('newbalanceOrig').value),
            oldbalanceDest: parseFloat(document.getElementById('oldbalanceDest').value),
            newbalanceDest: parseFloat(document.getElementById('newbalanceDest').value)
        };

        try {
            const response = await fetch('http://127.0.0.1:8000/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(formData)
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            
            setTimeout(() => {
                loadingState.classList.add('hidden');
                showResult(data);
            }, 800); // Artificial delay for effect

        } catch (error) {
            console.error('Error:', error);
            loadingState.classList.add('hidden');
            errorState.classList.remove('hidden');
            errorText.textContent = "CONNECTION BREACHED. API ENDPOINT OFFLINE or CORS error.";
        }
    });

    function showResult(data) {
        resultDisplay.classList.remove('hidden');
        
        const probPercentage = (data.fraud_probability * 100).toFixed(2);
        const label = data.risk_label.toUpperCase();
        
        // Reset classes
        riskLabel.className = 'risk-badge';
        probFill.className = 'progress-fill';
        
        // Add new classes based on risk
        if (label === 'HIGH') {
            riskLabel.classList.add('high');
            probFill.classList.add('high');
        } else if (label === 'MEDIUM') {
            riskLabel.classList.add('medium');
            probFill.classList.add('medium');
        } else {
            riskLabel.classList.add('low');
            probFill.classList.add('low');
        }
        
        riskLabel.textContent = label;
        
        // Animate numbers
        let start = 0;
        const end = parseFloat(probPercentage);
        const duration = 1000;
        const startTime = performance.now();
        
        function updateNumber(currentTime) {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / duration, 1);
            
            const currentProb = (progress * end).toFixed(2);
            fraudProb.textContent = currentProb;
            
            if (progress < 1) {
                requestAnimationFrame(updateNumber);
            }
        }
        
        requestAnimationFrame(updateNumber);
        
        // Animate bar
        setTimeout(() => {
            probFill.style.width = `${probPercentage}%`;
        }, 100);
    }
});
