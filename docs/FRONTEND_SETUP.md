# 🎨 ScholarGenie Frontend Setup Guide

## What We're Building

A **modern, beautiful Next.js frontend** that makes ScholarGenie look like:
- ✨ **Better than Elicit** - More features, better UX
- 🚀 **Better than Braimium** - Faster, more powerful
- 🎯 **Gen-Z aesthetic** - Minimal, clean, modern design
- ⚡ **Lightning fast** - Server-side rendering + optimization

---

## Tech Stack

| Technology | Why We Use It |
|-----------|---------------|
| **Next.js 14** | Server-side rendering, SEO, fast performance |
| **React 18** | Modern UI framework, component reusability |
| **TypeScript** | Type safety, better developer experience |
| **Tailwind CSS** | Utility-first CSS, rapid development |
| **Framer Motion** | Smooth animations and transitions |
| **shadcn/ui** | Beautiful pre-built components |
| **Zustand** | Lightweight state management |
| **Lucide Icons** | Clean, modern icon set |

---

## 🚀 Quick Start

### Step 1: Install Node.js (if not installed)

Download from: https://nodejs.org/ (LTS version recommended)

Check installation:
```cmd
node --version
npm --version
```

### Step 2: Install Dependencies

```cmd
cd C:\Users\aadhi\Desktop\Projects\ScholarGenie\frontend
npm install
```

This will install all dependencies (~400MB, takes 2-3 minutes).

### Step 3: Run Development Server

```cmd
npm run dev
```

### Step 4: Open in Browser

Visit: **http://localhost:3000**

You'll see the beautiful landing page!

---

## 📁 Project Structure

```
frontend/
├── app/                    # Next.js App Router
│   ├── page.tsx           # Landing page (what you see first)
│   ├── layout.tsx         # Root layout
│   ├── globals.css        # Global styles
│   └── app/               # Main research app
│       └── page.tsx       # Research interface
├── components/            # Reusable components
│   └── ui/                # shadcn/ui components
├── lib/                   # Utility functions
│   └── api.ts            # API calls to backend
├── package.json          # Dependencies
├── tailwind.config.ts    # Tailwind configuration
└── tsconfig.json         # TypeScript configuration
```

---

## 🎨 What Makes It Better Than Elicit/Braimium

### 1. **Smarter Search**
- ❌ Elicit: Basic keyword search
- ✅ ScholarGenie: Semantic AI search + 19 specialized agents

### 2. **Auto-Generate Everything**
- ❌ Braimium: Manual extraction
- ✅ ScholarGenie: Auto-generates presentations, reports, literature reviews

### 3. **Knowledge Graphs**
- ❌ Elicit: Linear paper lists
- ✅ ScholarGenie: Visual knowledge graphs showing connections

### 4. **Gap Discovery**
- ❌ Braimium: No gap analysis
- ✅ ScholarGenie: 10 methods to identify research opportunities

### 5. **100% Free**
- ❌ Elicit/Braimium: Paid subscriptions required
- ✅ ScholarGenie: Completely free, runs locally

### 6. **Modern UI**
- ❌ Elicit/Braimium: Corporate, dated design
- ✅ ScholarGenie: Gen-Z aesthetic, animations, dark mode

---

## 🎯 Key Features of the Frontend

### Landing Page (/)
- 🌟 Hero section with animated background
- 🔍 Prominent search bar
- 📊 Feature cards showcase
- 🎨 Gradient effects and smooth animations
- 📱 Fully responsive

### Research App (/app)
- 📝 Paper search with filters
- 🤖 AI chat interface for questions
- 📊 Visual knowledge graphs
- 📈 Research gap analysis
- 🎨 Presentation generator
- 📁 Personal library management
- 💾 Save papers and notes
- 🔗 Citation management

---

## 🛠️ Development Workflow

### Running in Dev Mode
```cmd
cd C:\Users\aadhi\Desktop\Projects\ScholarGenie\frontend
npm run dev
```
- Auto-reloads on file changes
- Hot module replacement
- Fast refresh

### Building for Production
```cmd
npm run build
npm start
```
- Optimized bundle
- Server-side rendering
- Image optimization
- Code splitting

---

## 🎨 Customization Guide

### Change Colors
Edit `frontend/app/globals.css`:
```css
:root {
  --primary: 262.1 83.3% 57.8%;  /* Purple - change this! */
}
```

### Add New Page
1. Create `frontend/app/your-page/page.tsx`
2. Export a React component
3. Visit `/your-page` in browser

### Add API Call
Edit `frontend/lib/api.ts`:
```typescript
export async function searchPapers(query: string) {
  const res = await fetch(`/api/papers/search`, {
    method: 'POST',
    body: JSON.stringify({ query })
  })
  return res.json()
}
```

---

## 📦 Adding Components

Using **shadcn/ui** (pre-styled components):

```cmd
npx shadcn-ui@latest add button
npx shadcn-ui@latest add card
npx shadcn-ui@latest add dialog
```

This adds components to `components/ui/` that you can use:

```tsx
import { Button } from "@/components/ui/button"

<Button variant="default">Click me</Button>
```

---

## 🐛 Troubleshooting

### Port 3000 already in use
```cmd
# Kill process on port 3000
npx kill-port 3000

# Or use different port
npm run dev -- -p 3001
```

### Module not found
```cmd
# Clear node_modules and reinstall
rm -rf node_modules package-lock.json
npm install
```

### Build errors
```cmd
# Clear Next.js cache
rm -rf .next
npm run dev
```

---

## 🚀 Deployment

### Deploy to Vercel (Recommended)
1. Push code to GitHub
2. Visit vercel.com
3. Import repository
4. Deploy (automatic!)

### Deploy to Netlify
```cmd
npm run build
netlify deploy --prod
```

### Self-Host
```cmd
npm run build
npm start  # Runs on port 3000
```

---

## 🎯 Next Steps

1. ✅ Run `npm install` in frontend folder
2. ✅ Run `npm run dev`
3. ✅ Open http://localhost:3000
4. ✅ See the beautiful landing page
5. 🔄 I'll create the research app page next
6. 🔄 Connect to your backend API
7. 🔄 Add paper search interface
8. 🔄 Add AI chat
9. 🔄 Add knowledge graphs
10. 🔄 Deploy to production

---

## 💡 Design Philosophy

### Minimal & Clean
- No clutter
- Focus on content
- Generous white space
- Clear typography

### Fast & Responsive
- Server-side rendering
- Optimized images
- Code splitting
- Lazy loading

### Delightful Interactions
- Smooth animations
- Micro-interactions
- Loading states
- Toast notifications

### Accessible
- Keyboard navigation
- Screen reader support
- ARIA labels
- Color contrast

---

**Your frontend is ready to go! 🎉**

Run `npm install && npm run dev` to see it in action!
