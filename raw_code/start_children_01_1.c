static void start_children(FFStream *feed)
{
    if (no_launch)
        return;
    for (; feed; feed = feed->next) {
        if (feed->child_argv && !feed->pid) {
            feed->pid_start = time(0);
            feed->pid = fork();
            if (feed->pid < 0) {
                fprintf(stderr, "Unable to create children\n");
                exit(1);
            }
            if (!feed->pid) {
                /* In child */
                char pathname[1024];
                char *slash;
                int i;
                for (i = 3; i < 256; i++) {
                    close(i);
                }
                if (!ffserver_debug) {
                    i = open("/dev/null", O_RDWR);
                    if (i)
                        dup2(i, 0);
                    dup2(i, 1);
                    dup2(i, 2);
                    if (i)
                        close(i);
                }
                pstrcpy(pathname, sizeof(pathname), my_program_name);
                slash = strrchr(pathname, '/');
                if (!slash) {
                    slash = pathname;
                } else {
                    slash++;
                }
                strcpy(slash, "ffmpeg");
                /* This is needed to make relative pathnames work */
                chdir(my_program_dir);
                execvp(pathname, feed->child_argv);
                _exit(1);
            }
        }
    }
}