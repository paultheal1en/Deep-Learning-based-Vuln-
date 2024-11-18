static void termsig_handler(int signum)

{

    static int sigterm_reported;

    if (!sigterm_reported) {

        sigterm_reported = (write(sigterm_wfd, "", 1) == 1);

    }

}
