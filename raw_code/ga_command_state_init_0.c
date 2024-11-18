void ga_command_state_init(GAState *s, GACommandState *cs)

{

    if (vss_init(true)) {

        ga_command_state_add(cs, NULL, guest_fsfreeze_cleanup);

    }

}
