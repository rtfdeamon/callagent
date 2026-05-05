const puppeteer = require('puppeteer');

(async () => {
    const browser = await puppeteer.launch({
        headless: "new",
        args: ['--no-sandbox', '--disable-setuid-sandbox']
    });
    const page = await browser.newPage();

    try {
        console.log('Navigating to http://localhost:3003...');
        await page.goto('http://localhost:3003/', { waitUntil: 'networkidle2' });

        console.log('Filling login form...');
        await page.type('input[name="username"], input[type="text"], input#username', 'admin', { delay: 50 });
        await page.type('input[name="password"], input[type="password"], input#password', 'mmvsadmin', { delay: 50 });

        await page.keyboard.press('Enter');
        console.log('Submitted login. Waiting for navigation...');
        await page.waitForNavigation({ waitUntil: 'networkidle2' });

        console.log('Logged in successfully, taking screenshot...');
        await page.screenshot({ path: 'dashboard.png' });

        console.log('Looking for settings/Ollama config...');
        // Extract the DOM or click settings
        const html = await page.content();
        const fs = require('fs');
        fs.writeFileSync('dashboard.html', html);

        console.log('Saved dashboard.html for inspection');

    } catch (error) {
        console.error('Error:', error);
        await page.screenshot({ path: 'error.png' });
    } finally {
        await browser.close();
    }
})();
