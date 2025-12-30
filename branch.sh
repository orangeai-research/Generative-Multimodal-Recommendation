

# ⚠️ 注意：执行前请先修改下面的三行信息！
git filter-branch --env-filter '
# 1. 这里填你在 git log 里看到的那个“错误的小号邮箱”
WRONG_EMAIL="orangeai.team@outlook.com"

# 2. 这里填你想绑定的“大号用户名”
NEW_NAME="orangeheyue"

# 3. 这里填你想绑定的“大号邮箱”
NEW_EMAIL="orangeheyue@163.com"

# 下面的逻辑不用动
if [ "$GIT_COMMITTER_EMAIL" = "$WRONG_EMAIL" ]
then
    export GIT_COMMITTER_NAME="$NEW_NAME"
    export GIT_COMMITTER_EMAIL="$NEW_EMAIL"
fi
if [ "$GIT_AUTHOR_EMAIL" = "$WRONG_EMAIL" ]
then
    export GIT_AUTHOR_NAME="$NEW_NAME"
    export GIT_AUTHOR_EMAIL="$NEW_EMAIL"
fi
' --tag-name-filter cat -- --branches --tags