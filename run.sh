osascript -e 'tell application "Terminal" to do script "cd /Users/isha/Desktop/Courses/RL/RLCaR && /usr/local/bin/python3.9 main.py -rl reinforce -cs 10"' &
osascript -e 'tell application "Terminal" to do script "cd /Users/isha/Desktop/Courses/RL/RLCaR && /usr/local/bin/python3.9 main.py -rl reinforce -cs 20"' &
osascript -e 'tell application "Terminal" to do script "cd /Users/isha/Desktop/Courses/RL/RLCaR && /usr/local/bin/python3.9 main.py -rl reinforce -cs 50"' &
osascript -e 'tell application "Terminal" to do script "cd /Users/isha/Desktop/Courses/RL/RLCaR && /usr/local/bin/python3.9 main.py -rl reinforce -cs 75"' &
osascript -e 'tell application "Terminal" to do script "cd /Users/isha/Desktop/Courses/RL/RLCaR && /usr/local/bin/python3.9 main.py -rl reinforce -cs 100"' &


osascript -e 'tell application "Terminal" to do script "cd /Users/isha/Desktop/Courses/RL/RLCaR && /usr/local/bin/python3.9 main.py -rl actor_critic -fa tc -cs 10"' &
osascript -e 'tell application "Terminal" to do script "cd /Users/isha/Desktop/Courses/RL/RLCaR && /usr/local/bin/python3.9 main.py -rl actor_critic -fa tc -cs 20"' &
osascript -e 'tell application "Terminal" to do script "cd /Users/isha/Desktop/Courses/RL/RLCaR && /usr/local/bin/python3.9 main.py -rl actor_critic -fa tc -cs 50"' &
osascript -e 'tell application "Terminal" to do script "cd /Users/isha/Desktop/Courses/RL/RLCaR && /usr/local/bin/python3.9 main.py -rl actor_critic -fa tc -cs 75"' &
osascript -e 'tell application "Terminal" to do script "cd /Users/isha/Desktop/Courses/RL/RLCaR && /usr/local/bin/python3.9 main.py -rl actor_critic -fa tc -cs 100"' &


osascript -e 'tell application "Terminal" to do script "cd /Users/isha/Desktop/Courses/RL/RLCaR && /usr/local/bin/python3.9 main.py -rl actor_critic -fa linear -cs 20"' &
osascript -e 'tell application "Terminal" to do script "cd /Users/isha/Desktop/Courses/RL/RLCaR && /usr/local/bin/python3.9 main.py -rl actor_critic -fa linear -cs 50"' &


osascript -e 'tell application "Terminal" to do script "cd /Users/isha/Desktop/Courses/RL/RLCaR && /usr/local/bin/python3.9 main.py -rl n_step_sarsa -fa linear -cs 10 -n_steps 1"' &
osascript -e 'tell application "Terminal" to do script "cd /Users/isha/Desktop/Courses/RL/RLCaR && /usr/local/bin/python3.9 main.py -rl n_step_sarsa -fa linear -cs 20 -n_steps 1"' &
osascript -e 'tell application "Terminal" to do script "cd /Users/isha/Desktop/Courses/RL/RLCaR && /usr/local/bin/python3.9 main.py -rl n_step_sarsa -fa linear -cs 50 -n_steps 1"' &
osascript -e 'tell application "Terminal" to do script "cd /Users/isha/Desktop/Courses/RL/RLCaR && /usr/local/bin/python3.9 main.py -rl n_step_sarsa -fa linear -cs 100 -n_steps 1"' &