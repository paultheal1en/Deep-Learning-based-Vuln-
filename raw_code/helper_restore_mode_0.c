void helper_restore_mode (void)

{

    env->ps = (env->ps & ~0xC) | env->saved_mode;

}
