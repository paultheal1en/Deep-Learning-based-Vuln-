void runstate_set(RunState new_state)

{

    if (new_state >= RUN_STATE_MAX ||

        !runstate_valid_transitions[current_run_state][new_state]) {

        fprintf(stderr, "invalid runstate transition\n");

        abort();

    }



    current_run_state = new_state;

}
