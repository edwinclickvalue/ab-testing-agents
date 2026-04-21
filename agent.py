import asyncio
import base64
import json
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from playwright.async_api import async_playwright

load_dotenv()
client = OpenAI()

SYSTEM_PROMPT = """You are a website visitor trying to complete a goal.
Navigate the website naturally, like a real user would.
Use the available tools to interact with the page.
When you have completed the goal or determined it's impossible, call finish()."""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "click",
            "description": "Click an element on the page",
            "parameters": {
                "type": "object",
                "properties": {
                    "selector": {"type": "string", "description": "CSS selector or text to click"},
                },
                "required": ["selector"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "type_text",
            "description": "Type text into a focused input field",
            "parameters": {
                "type": "object",
                "properties": {
                    "selector": {"type": "string", "description": "CSS selector of input field"},
                    "text": {"type": "string", "description": "Text to type"},
                },
                "required": ["selector", "text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scroll",
            "description": "Scroll down the page",
            "parameters": {
                "type": "object",
                "properties": {
                    "pixels": {"type": "integer", "description": "Pixels to scroll down"},
                },
                "required": ["pixels"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_page_content",
            "description": "Get the current page HTML content to understand what's visible",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "Signal that the goal is completed or impossible",
            "parameters": {
                "type": "object",
                "properties": {
                    "success": {"type": "boolean", "description": "Whether goal was achieved"},
                    "reason": {"type": "string", "description": "Explanation of what happened"},
                    "friction_points": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of UX issues encountered",
                    },
                },
                "required": ["success", "reason", "friction_points"],
            },
        },
    },
]


@dataclass
class TestResult:
    success: bool
    steps: int
    time_seconds: float
    reason: str
    friction_points: list[str]
    screenshot_base64: str = ""
    step_screenshots: list[str] = field(default_factory=list)
    error: str = ""


async def _accept_cookies(page) -> None:
    clicked = await page.evaluate("""() => {
        const ACCEPT = [
            // NL
            'alles accepteren', 'accepteer alles', 'accepteer alle',
            'alle cookies accepteren', 'alle cookies toestaan', 'akkoord met alle',
            'accepteren', 'akkoord', 'toestaan', 'ja, ik accepteer', 'bevestigen',
            // EN
            'accept all', 'allow all', 'i agree to all', 'agree to all',
            'allow cookies', 'accept cookies', 'i agree', 'agree', 'allow', 'accept',
            'okay', 'got it',
            // DE
            'alle akzeptieren', 'alle annehmen', 'alle cookies akzeptieren',
            'zustimmen', 'akzeptieren', 'annehmen', 'einverstanden', 'ich stimme zu',
            'alle auswählen', 'weiter',
            // FR
            'tout accepter', 'accepter tout', 'accepter tous', "j'accepte",
            'accepter', "j'accepte tout", "d'accord", 'tout autoriser',
            'autoriser tout', 'autoriser',
            // ES
            'aceptar todo', 'aceptar todas', 'permitir todo', 'acepto',
            'aceptar', 'permitir', 'de acuerdo',
            // IT
            'accetta tutto', 'accetta tutti', 'consenti tutto', 'accetto',
            'accetta', 'consenti', 'sono d\'accordo',
            // PT
            'aceitar tudo', 'aceitar todos', 'concordo com tudo',
            'aceitar', 'concordo', 'permitir tudo', 'permitir',
            // PL
            'zaakceptuj wszystko', 'akceptuj wszystko', 'zaakceptuj wszystkie',
            'zaakceptuj', 'akceptuję', 'akceptuj', 'zgadzam się',
            // CZ
            'přijmout vše', 'přijmout všechny', 'souhlasím se vším',
            'přijmout', 'souhlasím', 'přijímám',
            // SK
            'prijať všetko', 'prijať všetky', 'súhlasím',
            'prijať', 'akceptovať',
            // HU
            'összes elfogadása', 'mindent elfogad', 'elfogadom az összeset',
            'elfogad', 'elfogadom', 'beleegyezem',
            // RO
            'acceptați toate', 'acceptați tot', 'accept toate',
            'acceptați', 'accept', 'sunt de acord',
            // SV
            'acceptera alla', 'godkänn alla', 'godkänn allt',
            'acceptera', 'godkänn', 'jag godkänner',
            // DA
            'accepter alle', 'godkend alle', 'accepter alle cookies',
            'accepter', 'godkend', 'jeg accepterer',
            // NO
            'godta alle', 'aksepter alle', 'godta alle informasjonskapsler',
            'godta', 'aksepter', 'jeg godtar',
            // FI
            'hyväksy kaikki', 'hyväksy kaikki evästeet',
            'hyväksy', 'suostun',
            // EL (Greek)
            'αποδοχή όλων', 'αποδοχή όλων των cookies',
            'αποδοχή', 'συμφωνώ',
            // HR
            'prihvati sve', 'prihvati sve kolačiće', 'prihvati',
            // BG
            'приемане на всички', 'приемам всички', 'приемам',
            // ET
            'nõustu kõigiga', 'nõustu kõigi küpsistega', 'nõustu',
            // LV
            'piekrist visiem', 'piekrist', 'piekrītu',
            // LT
            'sutikti su visais', 'sutikti', 'sutinku',
            // SL
            'sprejmi vse', 'sprejmi', 'strinjam se',
            // MT
            'aċċetta kollox', 'aċċetta',
        ];
        const REJECT = [
            'policy', 'statement', 'verklaring', 'meer informatie', 'more info',
            'lees meer', 'read more', 'privacy policy', 'cookie policy',
            'instellingen', 'settings', 'aanpassen', 'customize', 'weigeren',
            'decline', 'reject',
        ];

        function score(el) {
            const txt = el.innerText.trim().toLowerCase();
            if (!txt || txt.length > 60) return 0;
            if (REJECT.some(r => txt.includes(r))) return 0;
            const matchIdx = ACCEPT.findIndex(a => txt.includes(a));
            if (matchIdx === -1) return 0;
            // higher score = earlier in ACCEPT list + prefer button over link
            const tagBonus = el.tagName === 'BUTTON' ? 100 : 0;
            return (ACCEPT.length - matchIdx) + tagBonus;
        }

        const candidates = Array.from(
            document.querySelectorAll('button, a, [role="button"]')
        ).filter(el => {
            const rect = el.getBoundingClientRect();
            return rect.width > 0 && rect.height > 0;  // visible
        });

        const scored = candidates
            .map(el => ({ el, s: score(el) }))
            .filter(x => x.s > 0)
            .sort((a, b) => b.s - a.s);

        if (scored.length > 0) {
            scored[0].el.click();
            return true;
        }
        return false;
    }""")
    if clicked:
        await page.wait_for_timeout(600)


async def run_agent_test(url: str, goal: str, persona: str = "casual shopper") -> TestResult:
    import time

    start = time.time()
    steps = 0
    screenshots_dir = Path("screenshots")
    screenshots_dir.mkdir(exist_ok=True)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": 1280, "height": 720})
        await page.goto(url, wait_until="networkidle")
        await _accept_cookies(page)

        messages = [
            {
                "role": "user",
                "content": f"You are a {persona}. Your goal: {goal}\n\nYou are on: {url}\nStart navigating to complete your goal.",
            }
        ]

        finish_result = None
        step_screenshots = []

        for _ in range(20):  # max 20 steps
            await page.wait_for_timeout(800)  # allow JS-rendered banners to appear
            await _accept_cookies(page)
            screenshot = await page.screenshot(type="jpeg", quality=70, full_page=False)
            screenshot_b64 = base64.b64encode(screenshot).decode()
            step_screenshots.append(screenshot_b64)

            links = await page.evaluate("""() => {
                return Array.from(document.querySelectorAll('a, button, [role=button]'))
                    .map(el => ({ text: el.innerText.trim(), tag: el.tagName, href: el.href || '' }))
                    .filter(el => el.text)
                    .slice(0, 60);
            }""")
            page_text = await page.inner_text("body")
            messages.append({
                "role": "user",
                "content": f"CLICKABLE ELEMENTS (use exact text to click):\n{json.dumps(links, indent=2)}\n\nPAGE TEXT:\n{page_text[:2000]}",
            })

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "system", "content": SYSTEM_PROMPT}] + messages,
                tools=TOOLS,
                tool_choice="required",
            )

            msg = response.choices[0].message
            messages.append(msg)

            if not msg.tool_calls:
                break

            tool_results = []
            for tool_call in msg.tool_calls:
                name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)
                steps += 1

                if name == "finish":
                    finish_result = args
                    tool_results.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "content": "Done.",
                    })
                elif name == "click":
                    selector = args["selector"]
                    clicked = False
                    last_err = None
                    for attempt in [
                        lambda: page.click(selector, timeout=8000),
                        lambda: page.get_by_text(selector, exact=False).first.click(timeout=8000),
                        lambda: page.locator(f"a:has-text('{selector}')").first.click(timeout=8000),
                        lambda: page.locator(f"button:has-text('{selector}')").first.click(timeout=8000),
                    ]:
                        try:
                            await attempt()
                            clicked = True
                            break
                        except Exception as e:
                            last_err = e
                    if clicked:
                        try:
                            await page.wait_for_load_state("networkidle", timeout=5000)
                        except Exception:
                            pass
                        result = f"Clicked: {selector}"
                    else:
                        result = f"Click failed for '{selector}': {last_err}"
                    tool_results.append({"tool_call_id": tool_call.id, "role": "tool", "content": result})
                elif name == "type_text":
                    try:
                        await page.fill(args["selector"], args["text"])
                        result = f"Typed into {args['selector']}"
                    except Exception as e:
                        result = f"Type failed: {e}"
                    tool_results.append({"tool_call_id": tool_call.id, "role": "tool", "content": result})
                elif name == "scroll":
                    await page.evaluate(f"window.scrollBy(0, {args['pixels']})")
                    tool_results.append({"tool_call_id": tool_call.id, "role": "tool", "content": f"Scrolled {args['pixels']}px"})
                elif name == "get_page_content":
                    links = await page.evaluate("""() => {
                        return Array.from(document.querySelectorAll('a, button, [role=button]'))
                            .map(el => ({ text: el.innerText.trim(), tag: el.tagName, href: el.href || '' }))
                            .filter(el => el.text)
                            .slice(0, 60);
                    }""")
                    text = await page.inner_text("body")
                    content = f"CLICKABLE ELEMENTS:\n{json.dumps(links, indent=2)}\n\nPAGE TEXT:\n{text[:3000]}"
                    tool_results.append({"tool_call_id": tool_call.id, "role": "tool", "content": content})

            messages.extend(tool_results)

            if finish_result:
                break

        final_screenshot = await page.screenshot(type="jpeg", quality=70, full_page=False)
        final_b64 = base64.b64encode(final_screenshot).decode()
        if not step_screenshots or step_screenshots[-1] != final_b64:
            step_screenshots.append(final_b64)
        await browser.close()

    elapsed = time.time() - start

    if finish_result:
        return TestResult(
            success=finish_result["success"],
            steps=steps,
            time_seconds=round(elapsed, 2),
            reason=finish_result["reason"],
            friction_points=finish_result["friction_points"],
            screenshot_base64=final_b64,
            step_screenshots=step_screenshots,
        )

    return TestResult(
        success=False,
        steps=steps,
        time_seconds=round(elapsed, 2),
        reason="Max steps reached without completing goal",
        friction_points=["Goal not reached within step limit"],
        screenshot_base64=final_b64,
        step_screenshots=step_screenshots,
    )
