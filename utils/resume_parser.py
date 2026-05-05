import os
import re
import json

# Try pdfminer first (better extraction), fallback to PyPDF2
try:
    from pdfminer.high_level import extract_text as pdfminer_extract
    HAS_PDFMINER = True
except ImportError:
    HAS_PDFMINER = False

try:
    import PyPDF2
    HAS_PYPDF2 = True
except ImportError:
    HAS_PYPDF2 = False

try:
    import docx2txt
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False

# Anthropic client for LLM-based extraction. Cleaner than regex — understands
# context, picks up modern skills not in any hardcoded list, handles odd
# formatting. Falls back to the regex parser below if the API call fails.
# Defensive: catch ANY exception during setup (missing package, invalid key,
# network error during init) so module import never fails.
_claude = None
try:
    from anthropic import Anthropic
    _api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("Anthropic_API_KEY")
    if _api_key:
        _claude = Anthropic(api_key=_api_key)
        print("✅ Resume parser: Claude SDK initialized")
    else:
        print("⚠️  Resume parser: ANTHROPIC_API_KEY not set — will use regex fallback only")
except Exception as e:
    print(f"⚠️  Resume parser: Claude SDK init failed ({e}) — will use regex fallback only")
    _claude = None


SKILLS_DB = [
    # Frontend
    "html", "html5", "css", "css3", "sass", "scss", "tailwind", "tailwind css",
    "bootstrap", "material ui", "chakra ui",
    "javascript", "typescript", "jquery", "react", "react.js", "reactjs",
    "next.js", "nextjs", "angular", "angularjs", "vue", "vue.js", "vuejs",
    "svelte", "gatsby", "webpack", "vite",
    # Backend
    "node", "node.js", "nodejs", "express", "express.js",
    "python", "flask", "django", "fastapi",
    "java", "spring", "spring boot", "hibernate",
    "c#", ".net", "asp.net",
    "php", "laravel", "ruby", "ruby on rails",
    "go", "golang", "rust", "kotlin", "scala",
    # Mobile
    "react native", "flutter", "dart", "swift", "android", "ios",
    # Databases
    "sql", "mysql", "postgresql", "postgres", "sqlite",
    "mongodb", "mongoose", "firebase", "firestore",
    "redis", "cassandra", "dynamodb", "neo4j", "graphql", "prisma",
    # Cloud & DevOps
    "aws", "amazon web services", "azure", "gcp", "google cloud",
    "heroku", "vercel", "netlify", "render",
    "docker", "kubernetes", "jenkins", "ci/cd",
    "terraform", "ansible", "nginx", "linux",
    "git", "github", "gitlab",
    # AI/ML
    "machine learning", "deep learning", "artificial intelligence",
    "tensorflow", "pytorch", "keras", "scikit-learn",
    "pandas", "numpy", "matplotlib",
    "nlp", "computer vision", "opencv",
    "data science", "data analysis", "power bi", "tableau",
    # Testing
    "jest", "cypress", "selenium", "postman",
    # Others
    "rest api", "microservices", "websocket", "socket.io",
    "agile", "scrum", "jira", "figma",
    "blockchain", "solidity", "web3",
    "c", "c++", "data structures", "algorithms",
    "oops", "full stack", "full-stack", "mern", "mean",
]

SECTION_HEADERS = [
    "experience", "work experience", "professional experience", "employment",
    "education", "academic", "qualifications",
    "projects", "personal projects", "academic projects",
    "skills", "technical skills", "core skills", "key skills",
    "interests", "hobbies", "languages",
    "declaration",
    "career objective", "objective", "summary", "professional summary",
    "courses", "coursework", "training",
    "certificates", "certifications", "awards", "achievements",
    "publications", "research", "references",
    "internship", "internships",
]


def extract_text_from_file(file_path):
    """Extract text from PDF or DOCX"""
    ext = os.path.splitext(file_path)[1].lower()

    if ext == '.pdf':
        if HAS_PDFMINER:
            return pdfminer_extract(file_path)
        elif HAS_PYPDF2:
            text = ""
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted
            return text
        else:
            raise ImportError("No PDF parser available. Install pdfminer.six or PyPDF2.")

    elif ext in ['.docx', '.doc']:
        if HAS_DOCX:
            return docx2txt.process(file_path)
        else:
            raise ImportError("docx2txt not installed.")

    else:
        raise ValueError("Unsupported format. Upload .pdf or .docx")


def _extract_with_claude(text):
    """
    Use Claude haiku-4-5 to intelligently extract structured data from resume text.
    Returns the same dict shape as the regex parser, or None on failure.
    Far more accurate than regex/SKILLS_DB matching — understands context,
    picks up modern tech, handles unusual formatting and resume layouts.
    """
    if not _claude or not text:
        return None

    # Cap input to ~12k chars to keep cost predictable (resumes rarely exceed this)
    snippet = text[:12000]

    system_prompt = """You are a precise resume parser. Extract structured data from the resume text and return ONLY a JSON object — no preamble, no markdown fences.

Required fields:
  - name: full name of the candidate (string or null if not found)
  - email: primary email address (string or null)
  - mobile_number: phone number, digits only, may include leading +country code (string, may be empty "")
  - skills: array of technical skills mentioned (programming languages, frameworks, tools, databases, cloud, etc.). Include ALL skills found, not just popular ones. Capitalize properly (e.g. "Python", "React.js", "AWS"). Limit to most relevant 25.
  - job_role: best inferred role from the resume (e.g. "Full Stack Developer", "Data Scientist", "Software Engineer"). String.
  - sector: industry sector (e.g. "Information Technology", "Data & Analytics", "Finance"). String.
  - linkedin: LinkedIn profile URL if present (string or null)
  - github: GitHub profile URL if present (string or null)

Rules:
  - Skills must be ACTUAL technologies or tools, not soft skills like "teamwork".
  - If a field is genuinely not present, return null (or "" for mobile_number, [] for skills).
  - Do NOT invent or guess. Better null than wrong.
  - Output strict JSON only."""

    try:
        response = _claude.messages.create(
            model="claude-haiku-4-5",
            max_tokens=1024,
            temperature=0.1,
            system=[{
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[{"role": "user", "content": f"Resume text:\n\n{snippet}"}],
        )
        raw = next((b.text for b in response.content if b.type == "text"), "")
        # Strip markdown fences if Claude added them despite instructions
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```", 2)[1]
            if raw.lstrip().lower().startswith("json"):
                raw = raw.lstrip()[4:]
            raw = raw.strip()

        parsed = json.loads(raw)
        # Normalize fields to expected types
        return {
            "name":          parsed.get("name") or None,
            "email":         parsed.get("email") or None,
            "mobile_number": parsed.get("mobile_number") or "",
            "skills":        parsed.get("skills") if isinstance(parsed.get("skills"), list) else [],
            "job_role":      parsed.get("job_role") or "Software Developer",
            "sector":        parsed.get("sector") or "Information Technology",
            "linkedin":      parsed.get("linkedin") or None,
            "github":        parsed.get("github") or None,
        }
    except Exception as e:
        print(f"⚠️  Claude resume extraction failed (will fall back to regex): {e}")
        return None


def extract_data_from_pdf(file_path):
    """
    Production resume parser.
    Returns: name, email, mobile_number, skills, job_role, sector, linkedin, github

    Strategy:
      1. Extract raw text from PDF/DOCX (pdfminer.six / PyPDF2 / docx2txt)
      2. Try Claude haiku-4-5 first — semantic extraction, picks up everything
      3. Fall back to the regex/SKILLS_DB parser if Claude is unavailable or
         returns junk. Two layers of defense → resume parsing always returns
         SOMETHING usable.
    """
    try:
        text = extract_text_from_file(file_path)

        if not text or len(text.strip()) < 20:
            return {"name": None, "email": None, "mobile_number": "", "skills": ["Python"], "job_role": "Developer", "sector": "IT"}

        # --- Try Claude first ---
        claude_result = _extract_with_claude(text)
        if claude_result and (claude_result.get("name") or claude_result.get("email") or claude_result.get("skills")):
            print(f"✅ Resume parsed via Claude: name={claude_result.get('name')}, "
                  f"skills={len(claude_result.get('skills') or [])}")
            return claude_result

        # --- Fallback: regex parser below ---
        print("⚠️  Claude unavailable or returned empty — using regex fallback")

        text_lower = text.lower()
        lines = [line.strip() for line in text.split('\n') if line.strip()]

        # --- Name (strict extraction) ---
        # Common non-name words found in resume headers/summaries
        NON_NAME_WORDS = {
            # Resume headers
            "resume", "curriculum", "vitae", "cv", "profile", "summary",
            "objective", "career", "seeking", "looking", "hardworking",
            "passionate", "motivated", "experienced", "fresher", "professional",
            "dedicated", "enthusiastic", "aspiring", "dynamic", "self",
            "driven", "detail", "oriented", "result", "results",
            # Contact labels
            "phone", "email", "address", "contact", "mobile", "tel",
            # Job titles
            "developer", "engineer", "student", "intern", "manager",
            "analyst", "designer", "architect", "consultant", "lead",
            # Indian cities (common in resumes)
            "hyderabad", "mumbai", "bangalore", "bengaluru", "chennai",
            "delhi", "kolkata", "pune", "ahmedabad", "jaipur",
            "lucknow", "kanpur", "nagpur", "indore", "bhopal",
            "visakhapatnam", "vizag", "coimbatore", "kochi", "cochin",
            "thiruvananthapuram", "mysore", "mysuru", "noida", "gurgaon",
            "gurugram", "chandigarh", "patna", "ranchi", "bhubaneswar",
            "guwahati", "surat", "vadodara", "rajkot", "mangalore",
            "mangaluru", "tirupati", "vijayawada", "guntur", "warangal",
            "nellore", "kakinada", "rajahmundry", "khammam", "nizamabad",
            "karimnagar", "anantapur", "kurnool", "kadapa", "ongole",
            "srikakulam", "eluru", "machilipatnam", "tenali", "proddatur",
            # Global cities
            "london", "new york", "san francisco", "seattle", "chicago",
            "toronto", "singapore", "dubai", "sydney", "melbourne",
            # Indian states
            "andhra", "pradesh", "telangana", "karnataka", "tamil", "nadu",
            "maharashtra", "kerala", "gujarat", "rajasthan", "uttar",
            "madhya", "west", "bengal", "odisha", "bihar", "jharkhand",
            # Address indicators
            "street", "road", "nagar", "colony", "sector", "phase",
            "block", "flat", "floor", "house", "plot", "lane",
            "area", "district", "state", "pin", "pincode", "zip",
            "india", "country",
        }

        def is_valid_name(candidate):
            """Check if a candidate string looks like a person's name, not a place/title"""
            if not candidate or len(candidate) < 2:
                return False
            words = candidate.split()
            if len(words) < 1 or len(words) > 3:
                return False
            word_lower = set(w.lower() for w in words)
            # Reject if any word is in blocklist
            if word_lower.intersection(NON_NAME_WORDS):
                return False
            # Reject if any word is too short (single letter except initials like "K" or "S")
            if any(len(w) == 1 and not w.isupper() for w in words):
                return False
            # All words must start with uppercase
            if not all(w[0].isupper() for w in words if w):
                return False
            # Reject single words that are very common non-names
            if len(words) == 1 and len(words[0]) <= 3:
                return False
            return True

        name = None
        # Strategy 1: Find name near email — name is usually right above/before email
        email_match_temp = re.search(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', text)
        if email_match_temp:
            email_pos = text.find(email_match_temp.group(0))
            text_before_email = text[:email_pos]
            before_lines = [l.strip() for l in text_before_email.split('\n') if l.strip()]
            for line in reversed(before_lines[-3:]):
                # Skip lines with numbers, special chars (address/phone lines)
                if re.search(r'\d{3,}|[,;|•\-–]', line):
                    continue
                candidate = re.sub(r'[^a-zA-Z\s.]', '', line).strip()
                if is_valid_name(candidate):
                    name = candidate
                    break

        # Strategy 2: Fallback — scan first 8 lines
        if not name:
            for line in lines[:8]:
                lc = line.strip()
                if not lc or len(lc) < 2:
                    continue
                # Skip lines with numbers, special chars, URLs
                if re.search(r'[@\d:,/\\|(){}#•\-–—]', lc):
                    continue
                # Skip section headers
                if any(lc.lower().strip() == h or lc.lower().startswith(h + ":") for h in SECTION_HEADERS):
                    continue
                candidate = re.sub(r'[^a-zA-Z\s.]', '', lc).strip()
                if is_valid_name(candidate):
                    name = candidate
                    if name and len(name) > 1:
                        break

        # --- Email ---
        email_match = re.search(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', text)
        email = email_match.group(0) if email_match else None

        # --- Phone ---
        phone = None
        for pattern in [r'(?:\+91[\s.-]?)?[6-9]\d{4}[\s.-]?\d{5}', r'(?:\+?\d{1,3}[\s.-]?)?\(?\d{2,4}\)?[\s.-]?\d{3,4}[\s.-]?\d{4}']:
            match = re.search(pattern, text)
            if match:
                phone = re.sub(r'[^\d+]', '', match.group(0))
                break

        # --- Skills ---
        found_skills = set()
        for skill in SKILLS_DB:
            if len(skill) <= 3:
                if re.search(r'\b' + re.escape(skill) + r'\b', text_lower):
                    found_skills.add(skill.upper() if len(skill) <= 2 else skill.title())
            else:
                if skill in text_lower:
                    display = skill
                    if skill.endswith('.js'):
                        display = skill
                    else:
                        display = skill.title()
                    found_skills.add(display)

        # --- LinkedIn & GitHub ---
        linkedin = None
        github = None
        for url in re.findall(r'https?://[^\s,)]+', text):
            ul = url.lower()
            if 'linkedin.com' in ul and not linkedin:
                linkedin = url.rstrip('.')
            if 'github.com' in ul and not github:
                github = url.rstrip('.')

        # --- Job Role ---
        role_map = {
            "Full Stack Developer": ["full stack", "fullstack", "full-stack", "mern", "mean"],
            "Frontend Developer": ["frontend", "front end", "react developer", "ui developer"],
            "Backend Developer": ["backend", "back end", "server side"],
            "Data Scientist": ["data science", "data scientist"],
            "DevOps Engineer": ["devops", "dev ops", "sre"],
            "Mobile Developer": ["android developer", "ios developer", "flutter developer"],
            "ML Engineer": ["machine learning", "deep learning", "ai engineer"],
            "Software Engineer": ["software engineer", "software developer", "sde"],
        }
        job_role = "Software Developer"
        for role, kws in role_map.items():
            if any(kw in text_lower for kw in kws):
                job_role = role
                break

        # --- Sector ---
        sector_map = {
            "Information Technology": ["software", "developer", "engineer", "programming", "tech"],
            "Data & Analytics": ["data science", "analytics", "machine learning"],
            "Finance": ["finance", "banking", "fintech"],
            "Healthcare": ["healthcare", "medical", "pharma"],
        }
        sector = "Information Technology"
        for sec, kws in sector_map.items():
            if any(kw in text_lower for kw in kws):
                sector = sec
                break

        return {
            "name": name,
            "email": email,
            "mobile_number": phone or "",
            "skills": sorted(list(found_skills)) if found_skills else ["Python", "Problem Solving"],
            "job_role": job_role,
            "sector": sector,
            "linkedin": linkedin,
            "github": github,
        }

    except Exception as e:
        print(f"Resume parsing error: {e}")
        return {
            "name": None, "email": None, "mobile_number": "",
            "skills": ["Python", "Problem Solving"],
            "job_role": "Developer", "sector": "Information Technology",
            "linkedin": None, "github": None,
        }
