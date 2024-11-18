bool ftrace_init(void)

{

    char mount_point[PATH_MAX];

    char path[PATH_MAX];

    int debugfs_found;

    int trace_fd = -1;



    debugfs_found = find_mount(mount_point, "debugfs");

    if (debugfs_found) {

        snprintf(path, PATH_MAX, "%s/tracing/tracing_on", mount_point);

        trace_fd = open(path, O_WRONLY);

        if (trace_fd < 0) {

            if (errno == EACCES) {

                trace_marker_fd = open("/dev/null", O_WRONLY);

                if (trace_marker_fd != -1) {

                    return true;

                }

            }

            perror("Could not open ftrace 'tracing_on' file");

            return false;

        } else {

            if (write(trace_fd, "1", 1) < 0) {

                perror("Could not write to 'tracing_on' file");

                close(trace_fd);

                return false;

            }

            close(trace_fd);

        }

        snprintf(path, PATH_MAX, "%s/tracing/trace_marker", mount_point);

        trace_marker_fd = open(path, O_WRONLY);

        if (trace_marker_fd < 0) {

            perror("Could not open ftrace 'trace_marker' file");

            return false;

        }

    } else {

        fprintf(stderr, "debugfs is not mounted\n");

        return false;

    }



    return true;

}
