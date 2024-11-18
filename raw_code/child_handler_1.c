static void child_handler(int sig)

{

    int status;

    while (waitpid(-1, &status, WNOHANG) > 0) /* NOTHING */;

}
