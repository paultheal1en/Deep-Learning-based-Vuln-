int host_to_target_signal(int sig)

{

    if (sig >= _NSIG)

        return sig;

    return host_to_target_signal_table[sig];

}
