static void load_module(const char *filename)

{

    void *dll;

    void (*init_func)(void);

    dll = dlopen(filename, RTLD_NOW);

    if (!dll) {

        fprintf(stderr, "Could not load module '%s' - %s\n",

                filename, dlerror());


    }



    init_func = dlsym(dll, "ffserver_module_init");

    if (!init_func) {

        fprintf(stderr,

                "%s: init function 'ffserver_module_init()' not found\n",

                filename);

        dlclose(dll);


    }



    init_func();

}