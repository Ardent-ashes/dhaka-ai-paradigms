%% ai-lab-report.cls -- Artificial Intelligence Lab Report class
%% Compile with pdfLaTeX
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{ai-lab-report}[2026/07/11 AI Lab Report class]

\LoadClass[11pt,a4paper]{article}

% Page, font, language, and typography
\RequirePackage[a4paper,margin=1in,headheight=14pt]{geometry}
\RequirePackage[utf8]{inputenc}
\RequirePackage[T1]{fontenc}
\RequirePackage{lmodern}
\RequirePackage[english]{babel}
\RequirePackage{microtype}
\RequirePackage{setspace}
\RequirePackage{parskip}

% Mathematics, tables, figures, and colour
\RequirePackage{amsmath,amssymb,amsthm,mathtools,bm}
% Load siunitx when available; this keeps the template portable on basic TeX installs.
\IfFileExists{siunitx.sty}{\RequirePackage{siunitx}}{}
\RequirePackage[table,dvipsnames]{xcolor}
\RequirePackage{graphicx}
\RequirePackage{booktabs,tabularx,array,multirow,longtable}
\RequirePackage{caption,subcaption}
\RequirePackage{float,placeins}
\RequirePackage{tikz}
\usetikzlibrary{arrows.meta,positioning,shapes.geometric}

% Algorithms and source code
\IfFileExists{algorithm.sty}{\RequirePackage{algorithm}}{}
\IfFileExists{algpseudocode.sty}{\RequirePackage{algpseudocode}}{}
\RequirePackage{listings}
\lstdefinestyle{labcode}{
  basicstyle=\ttfamily\small,
  breaklines=true,
  frame=single,
  numbers=left,
  numberstyle=\tiny\color{gray},
  keywordstyle=\color{NavyBlue}\bfseries,
  commentstyle=\color{ForestGreen},
  stringstyle=\color{BrickRed},
  showstringspaces=false,
  columns=fullflexible
}
\lstset{style=labcode}

% Lists, headings, references, and links
\RequirePackage{enumitem}
\setlist{nosep,leftmargin=*}
\RequirePackage{titlesec}
\titleformat{\section}{\large\bfseries\color{MidnightBlue}}{\thesection.}{0.5em}{}
\titleformat{\subsection}{\normalsize\bfseries}{\thesubsection.}{0.5em}{}
\titlespacing*{\section}{0pt}{1.2em}{0.45em}
\titlespacing*{\subsection}{0pt}{0.85em}{0.3em}
\RequirePackage[numbers,sort&compress]{natbib}
\RequirePackage{csquotes}
\RequirePackage{xurl}
\RequirePackage[colorlinks=true,linkcolor=MidnightBlue,citecolor=MidnightBlue,urlcolor=MidnightBlue]{hyperref}
\RequirePackage[nameinlink,noabbrev]{cleveref}

% Header and footer
\RequirePackage{fancyhdr}
\pagestyle{fancy}
\fancyhf{}
\lhead{}
\rhead{\thepage}
\renewcommand{\headrulewidth}{0.4pt}
\renewcommand{\footrulewidth}{0pt}

% One-column, paper-like title block
\renewcommand{\maketitle}{%
  \begin{center}
    {\LARGE\bfseries \@title\par}
    \vspace{0.65em}
    {\large \@author\par}
    \vspace{0.2em}
    {\normalsize\@date\par}
  \end{center}
  \vspace{0.55em}\noindent\rule{\textwidth}{0.7pt}\vspace{0.55em}
}
\newcommand{\reportemail}[1]{\gdef\@reportemail{#1}}
\reportemail{}
\renewcommand{\author}[1]{\gdef\@author{#1\\[-0.1em]\normalsize\texttt{\@reportemail}}}
\newcommand{\assignmenttitle}[1]{\textbf{Assignment focus:} #1\par\medskip}
\newcommand{\keywords}[1]{\noindent\textbf{Keywords:} #1}
\newcommand{\placeholder}[1]{\textcolor{gray}{\textit{[#1]}}}
\newcommand{\code}[1]{\texttt{#1}}
\renewcommand{\abstractname}{Abstract}
\renewenvironment{abstract}{\small\noindent\textbf{Abstract.}\ }{\par\normalsize\vspace{0.4em}}
\setlength{\columnsep}{0.25in}
\onecolumn
\endinput
