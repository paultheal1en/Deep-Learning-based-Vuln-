sigterm_handler(int sig)

{

    received_sigterm = sig;

    received_nb_signals++;

    term_exit_sigsafe();

    if(received_nb_signals > 3)

        exit_program(123);

}
