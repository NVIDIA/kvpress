#!/bin/bash

# DCO Fix Script
# This script safely fixes DCO (Developer Certificate of Origin) issues by rebasing with signoff
# and ensures it works for current and future PRs

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== DCO Fix Script ===${NC}"

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in a git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    print_error "Not in a git repository!"
    exit 1
fi

# Get current branch
CURRENT_BRANCH=$(git branch --show-current)
print_info "Current branch: $CURRENT_BRANCH"

# Check if there are uncommitted changes
if ! git diff-index --quiet HEAD --; then
    print_error "You have uncommitted changes. Please commit or stash them first."
    git status --porcelain
    exit 1
fi

# Get git config
GIT_NAME=$(git config user.name)
GIT_EMAIL=$(git config user.email)

if [ -z "$GIT_NAME" ] || [ -z "$GIT_EMAIL" ]; then
    print_error "Git user.name and user.email must be configured!"
    print_info "Run: git config user.name 'Your Name'"
    print_info "Run: git config user.email 'your.email@example.com'"
    exit 1
fi

print_info "Git user: $GIT_NAME <$GIT_EMAIL>"

# Function to count commits ahead of main/master
count_commits_ahead() {
    # Try to find the base branch (main or master)
    if git show-ref --verify --quiet refs/remotes/origin/main; then
        BASE_BRANCH="origin/main"
    elif git show-ref --verify --quiet refs/remotes/origin/master; then
        BASE_BRANCH="origin/master"
    else
        print_warning "Could not find origin/main or origin/master. Trying local branches..."
        if git show-ref --verify --quiet refs/heads/main; then
            BASE_BRANCH="main"
        elif git show-ref --verify --quiet refs/heads/master; then
            BASE_BRANCH="master"
        else
            print_error "Could not find base branch (main/master). Please specify manually."
            return 1
        fi
    fi

    # Count commits ahead
    COMMITS_AHEAD=$(git rev-list --count ${BASE_BRANCH}..HEAD 2>/dev/null || echo "0")
    echo $COMMITS_AHEAD
}

# Allow manual specification of commit count
if [ "$1" = "--commits" ] && [ -n "$2" ]; then
    COMMIT_COUNT=$2
    print_info "Using manually specified commit count: $COMMIT_COUNT"
else
    # Auto-detect number of commits to rebase
    COMMIT_COUNT=$(count_commits_ahead)
    if [ "$COMMIT_COUNT" = "0" ]; then
        print_warning "Could not auto-detect commit count. Please specify manually:"
        print_info "Usage: $0 --commits <number_of_commits>"
        print_info "Or check the PR/branch to see how many commits you want to sign-off"
        exit 1
    fi
    print_info "Auto-detected commits ahead of base: $COMMIT_COUNT"
fi

# Confirm the operation
print_warning "This will rebase the last $COMMIT_COUNT commits and add sign-off to each."
print_warning "This will rewrite git history!"
print_info "Branch: $CURRENT_BRANCH"
print_info "Commits to rebase: $COMMIT_COUNT"

if [ "$1" != "--force" ]; then
    read -p "Do you want to continue? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "Operation cancelled."
        exit 0
    fi
fi

# Create a backup branch
BACKUP_BRANCH="${CURRENT_BRANCH}_backup_$(date +%Y%m%d_%H%M%S)"
print_info "Creating backup branch: $BACKUP_BRANCH"
git branch "$BACKUP_BRANCH"

# Set git editor to avoid interactive editors
export GIT_EDITOR="true"
export EDITOR="true"

# Perform the rebase with signoff
print_info "Starting rebase with signoff..."

# Use git rebase with signoff
if git rebase HEAD~${COMMIT_COUNT} --signoff --strategy-option=theirs; then
    print_success "Rebase with signoff completed successfully!"
else
    print_error "Rebase failed! Restoring from backup..."
    git rebase --abort 2>/dev/null || true
    git reset --hard "$BACKUP_BRANCH"
    git branch -D "$BACKUP_BRANCH" 2>/dev/null || true
    exit 1
fi

# Verify that commits now have sign-off
print_info "Verifying sign-off in recent commits..."
MISSING_SIGNOFF=0

for i in $(seq 1 $COMMIT_COUNT); do
    COMMIT_SHA=$(git rev-parse HEAD~$((i-1)))
    COMMIT_MSG=$(git log -1 --pretty=format:"%B" $COMMIT_SHA)

    if ! echo "$COMMIT_MSG" | grep -q "Signed-off-by: "; then
        print_warning "Commit $COMMIT_SHA still missing sign-off!"
        MISSING_SIGNOFF=$((MISSING_SIGNOFF + 1))
    fi
done

if [ $MISSING_SIGNOFF -eq 0 ]; then
    print_success "All commits now have proper sign-off!"
else
    print_warning "$MISSING_SIGNOFF commits still missing sign-off. You may need to fix them manually."
fi

# Show what changed
print_info "Recent commits with sign-off status:"
git log --oneline -${COMMIT_COUNT} --format="%h %s" | while read commit; do
    sha=$(echo $commit | cut -d' ' -f1)
    if git log -1 --pretty=format:"%B" $sha | grep -q "Signed-off-by: "; then
        echo -e "${GREEN}✓${NC} $commit"
    else
        echo -e "${RED}✗${NC} $commit"
    fi
done

print_info "Backup branch created: $BACKUP_BRANCH"
print_warning "To push the changes (rewriting history): git push --force-with-lease origin $CURRENT_BRANCH"
print_warning "To restore from backup if needed: git reset --hard $BACKUP_BRANCH"

echo -e "${GREEN}=== DCO Fix Complete ===${NC}"
