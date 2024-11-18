bool st_set_trace_file(const char *file)

{

    st_set_trace_file_enabled(false);



    free(trace_file_name);



    if (!file) {

        if (asprintf(&trace_file_name, CONFIG_TRACE_FILE, getpid()) < 0) {

            trace_file_name = NULL;

            return false;

        }

    } else {

        if (asprintf(&trace_file_name, "%s", file) < 0) {

            trace_file_name = NULL;

            return false;

        }

    }



    st_set_trace_file_enabled(true);

    return true;

}
