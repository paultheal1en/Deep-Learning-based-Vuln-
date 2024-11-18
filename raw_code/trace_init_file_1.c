void trace_init_file(const char *file)

{

#ifdef CONFIG_TRACE_SIMPLE

    st_set_trace_file(file);

#elif defined CONFIG_TRACE_LOG

    /* If both the simple and the log backends are enabled, "-trace file"

     * only applies to the simple backend; use "-D" for the log backend.

     */

    if (file) {

        qemu_set_log_filename(file);

    }

#else

    if (file) {

        fprintf(stderr, "error: -trace file=...: "

                "option not supported by the selected tracing backends\n");

        exit(1);

    }

#endif

}
