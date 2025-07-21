#!/bin/bash

# Define paths to the server scripts
BANANOMPY_SERVER="src/fastmcp/bananompy_server.py"
EOL_SERVER="src/fastmcp/eol_server.py"
CKAN_SERVER="src/fastmcp/ckan_server.py"

# Define process IDs for each server
BANANOMPY_PID=""
EOL_PID=""
CKAN_PID=""

# Start servers function
echo_start_servers() {
    echo "Starting all servers..."
    fastmcp run $BANANOMPY_SERVER &
    BANANOMPY_PID=$!
    echo "Bananompy server started with PID $BANANOMPY_PID"

    fastmcp run $EOL_SERVER &
    EOL_PID=$!
    echo "EOL server started with PID $EOL_PID"

    fastmcp run $CKAN_SERVER &
    CKAN_PID=$!
    echo "CKAN server started with PID $CKAN_PID"
}

# Stop servers function
echo_stop_servers() {
    echo "Stopping all servers..."
    kill $BANANOMPY_PID $EOL_PID $CKAN_PID
    echo "All servers stopped."
}

# Restart servers function
echo_restart_servers() {
    echo "Restarting all servers..."
    echo_stop_servers
    echo_start_servers
}

# Pause servers function
echo_pause_servers() {
    echo "Pausing all servers..."
    kill -STOP $BANANOMPY_PID $EOL_PID $CKAN_PID
    echo "All servers paused."
}

# Continue servers function
echo_continue_servers() {
    echo "Continuing all servers..."
    kill -CONT $BANANOMPY_PID $EOL_PID $CKAN_PID
    echo "All servers continued."
}

# Manage server functions
case "$1" in
    start)
        echo_start_servers
        ;;
    stop)
        echo_stop_servers
        ;;
    restart)
        echo_restart_servers
        ;;
    pause)
        echo_pause_servers
        ;;
    continue)
        echo_continue_servers
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|pause|continue}"
        exit 1
        ;;
esac

