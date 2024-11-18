static void __attribute__((constructor)) coroutine_init(void)

{

    if (!g_thread_supported()) {


        g_thread_init(NULL);





    }



    coroutine_cond = g_cond_new();

}