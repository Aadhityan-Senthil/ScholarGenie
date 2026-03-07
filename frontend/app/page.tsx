"use client"

import { useState } from "react"
import { useRouter } from "next/navigation"
import { motion } from "framer-motion"
import {
  Search,
  Sparkles,
  FileText,
  Presentation,
  Network,
  Zap,
  ArrowRight,
  Brain,
  BookOpen,
  TrendingUp
} from "lucide-react"
import Link from "next/link"

export default function Home() {
  const [searchQuery, setSearchQuery] = useState("")
  const router = useRouter()

  const handleSearch = () => {
    if (searchQuery.trim()) {
      router.push(`/app?q=${encodeURIComponent(searchQuery.trim())}`)
    } else {
      router.push("/app")
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-purple-950 to-slate-950">
      {/* Animated background grid */}
      <div className="absolute inset-0 bg-[linear-gradient(to_right,#4f4f4f2e_1px,transparent_1px),linear-gradient(to_bottom,#4f4f4f2e_1px,transparent_1px)] bg-[size:64px_64px] [mask-image:radial-gradient(ellipse_60%_50%_at_50%_0%,#000_70%,transparent_100%)]" />

      {/* Hero Section */}
      <div className="relative">
        {/* Navigation */}
        <nav className="container mx-auto px-6 py-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <Sparkles className="h-8 w-8 text-purple-500" />
              <span className="text-2xl font-bold gradient-text">ScholarGenie</span>
            </div>
            <div className="hidden md:flex items-center space-x-8">
              <Link href="/features" className="text-gray-300 hover:text-white transition">Features</Link>
              <Link href="/pricing" className="text-gray-300 hover:text-white transition">Pricing</Link>
              <Link href="/docs" className="text-gray-300 hover:text-white transition">Docs</Link>
              <Link
                href="/app"
                className="px-6 py-2 bg-purple-600 hover:bg-purple-700 rounded-full text-white transition font-medium"
              >
                Get Started
              </Link>
            </div>
          </div>
        </nav>

        {/* Hero Content */}
        <div className="container mx-auto px-6 pt-20 pb-32">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="text-center max-w-5xl mx-auto"
          >
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.2, duration: 0.8 }}
              className="inline-block mb-6"
            >
              <div className="flex items-center space-x-2 bg-purple-500/10 border border-purple-500/20 rounded-full px-4 py-2">
                <Zap className="h-4 w-4 text-purple-400" />
                <span className="text-sm text-purple-300">100% Free · No API Keys Required</span>
              </div>
            </motion.div>

            <h1 className="text-6xl md:text-8xl font-bold mb-6 leading-tight">
              Research{" "}
              <span className="gradient-text">
                10x Faster
              </span>
            </h1>

            <p className="text-xl md:text-2xl text-gray-300 mb-12 max-w-3xl mx-auto leading-relaxed">
              Your autonomous AI research assistant. Discover papers, analyze content,
              generate presentations, and uncover research gaps—all in seconds.
            </p>

            {/* Search Bar */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.4, duration: 0.8 }}
              className="max-w-3xl mx-auto"
            >
              <div className="relative group">
                <div className="absolute -inset-0.5 bg-gradient-to-r from-purple-600 to-pink-600 rounded-2xl blur opacity-30 group-hover:opacity-100 transition duration-1000 group-hover:duration-200" />
                <div className="relative flex items-center">
                  <input
                    type="text"
                    placeholder="What do you want to research today?"
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    onKeyDown={(e) => e.key === "Enter" && handleSearch()}
                    className="w-full px-8 py-6 bg-slate-900/90 backdrop-blur-xl border border-purple-500/20 rounded-2xl text-lg text-white placeholder-gray-400 focus:outline-none focus:border-purple-500/40 transition"
                  />
                  <button
                    onClick={handleSearch}
                    className="absolute right-3 px-6 py-3 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 rounded-xl text-white font-medium transition flex items-center space-x-2"
                  >
                    <span>Search</span>
                    <Search className="h-5 w-5" />
                  </button>
                </div>
              </div>

              <div className="flex items-center justify-center mt-6 space-x-4 text-sm text-gray-400">
                <span>Try:</span>
                {["Transformer models", "Quantum computing", "CRISPR gene editing"].map(s => (
                  <button
                    key={s}
                    onClick={() => router.push(`/app?q=${encodeURIComponent(s)}`)}
                    className="px-4 py-2 bg-slate-800/50 hover:bg-slate-800 rounded-lg transition"
                  >
                    {s}
                  </button>
                ))}
              </div>
            </motion.div>
          </motion.div>
        </div>
      </div>

      {/* Features Grid */}
      <div className="relative container mx-auto px-6 pb-32">
        <motion.div
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          transition={{ duration: 0.8 }}
          viewport={{ once: true }}
          className="grid md:grid-cols-3 gap-8"
        >
          <FeatureCard
            icon={<Brain className="h-8 w-8" />}
            title="AI-Powered Analysis"
            description="19 specialized AI agents work together to understand, summarize, and extract insights from papers"
            gradient="from-blue-500 to-cyan-500"
          />
          <FeatureCard
            icon={<Presentation className="h-8 w-8" />}
            title="Auto-Generate Slides"
            description="Create professional PowerPoint presentations from any paper in seconds"
            gradient="from-purple-500 to-pink-500"
          />
          <FeatureCard
            icon={<Network className="h-8 w-8" />}
            title="Knowledge Graphs"
            description="Visualize connections between papers, authors, and research topics automatically"
            gradient="from-green-500 to-emerald-500"
          />
          <FeatureCard
            icon={<TrendingUp className="h-8 w-8" />}
            title="Gap Discovery"
            description="Identify unexplored research opportunities and suggest novel directions"
            gradient="from-orange-500 to-red-500"
          />
          <FeatureCard
            icon={<FileText className="h-8 w-8" />}
            title="Literature Reviews"
            description="Synthesize multiple papers into comprehensive literature reviews"
            gradient="from-violet-500 to-purple-500"
          />
          <FeatureCard
            icon={<BookOpen className="h-8 w-8" />}
            title="Semantic Search"
            description="Find papers by meaning, not keywords. Ask questions in natural language"
            gradient="from-pink-500 to-rose-500"
          />
        </motion.div>
      </div>

      {/* CTA Section */}
      <div className="relative container mx-auto px-6 pb-32">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          viewport={{ once: true }}
          className="relative rounded-3xl overflow-hidden"
        >
          <div className="absolute inset-0 bg-gradient-to-r from-purple-600/20 to-pink-600/20" />
          <div className="relative px-12 py-16 text-center">
            <h2 className="text-4xl md:text-5xl font-bold mb-6">
              Ready to revolutionize your research?
            </h2>
            <p className="text-xl text-gray-300 mb-8 max-w-2xl mx-auto">
              Join researchers using ScholarGenie to accelerate their discoveries
            </p>
            <Link
              href="/app"
              className="inline-flex items-center space-x-2 px-8 py-4 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 rounded-full text-white text-lg font-medium transition"
            >
              <span>Start Researching</span>
              <ArrowRight className="h-5 w-5" />
            </Link>
          </div>
        </motion.div>
      </div>
    </div>
  )
}

function FeatureCard({
  icon,
  title,
  description,
  gradient
}: {
  icon: React.ReactNode
  title: string
  description: string
  gradient: string
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      whileInView={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      viewport={{ once: true }}
      whileHover={{ y: -8 }}
      className="relative group"
    >
      <div className="absolute -inset-0.5 bg-gradient-to-r opacity-0 group-hover:opacity-100 rounded-2xl blur transition duration-500" style={{ backgroundImage: `linear-gradient(to right, var(--tw-gradient-stops))` }} />
      <div className="relative h-full p-8 bg-slate-900/50 backdrop-blur-xl border border-slate-800 rounded-2xl">
        <div className={`inline-flex p-3 bg-gradient-to-r ${gradient} rounded-xl mb-4`}>
          {icon}
        </div>
        <h3 className="text-2xl font-bold mb-3">{title}</h3>
        <p className="text-gray-400 leading-relaxed">{description}</p>
      </div>
    </motion.div>
  )
}
