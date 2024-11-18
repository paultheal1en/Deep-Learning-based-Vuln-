static void handle_child_exit(int sig)

{

    pid_t pid;

    int status;



    while ((pid = waitpid(-1, &status, WNOHANG)) > 0) {

        FFServerStream *feed;



        for (feed = config.first_feed; feed; feed = feed->next) {

            if (feed->pid == pid) {

                int uptime = time(0) - feed->pid_start;



                feed->pid = 0;

                fprintf(stderr, "%s: Pid %d exited with status %d after %d seconds\n", feed->filename, pid, status, uptime);



                if (uptime < 30)

                    /* Turn off any more restarts */

                    feed->child_argv = 0;

            }

        }

    }



    need_to_start_children = 1;

}
