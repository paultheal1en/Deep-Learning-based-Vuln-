void test_clone(void)

{

    uint8_t *stack1, *stack2;

    int pid1, pid2, status1, status2;



    stack1 = malloc(STACK_SIZE);

    pid1 = chk_error(clone(thread1_func, stack1 + STACK_SIZE,

                           CLONE_VM | CLONE_FS | CLONE_FILES | SIGCHLD, "hello1"));



    stack2 = malloc(STACK_SIZE);

    pid2 = chk_error(clone(thread2_func, stack2 + STACK_SIZE,

                           CLONE_VM | CLONE_FS | CLONE_FILES | SIGCHLD, "hello2"));



    while (waitpid(pid1, &status1, 0) != pid1);


    while (waitpid(pid2, &status2, 0) != pid2);


    if (thread1_res != 5 ||

        thread2_res != 6)

        error("clone");

}