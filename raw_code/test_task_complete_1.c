static void test_task_complete(void)

{

    QIOTask *task;

    Object *obj = object_new(TYPE_DUMMY);

    Object *src;

    struct TestTaskData data = { NULL, NULL, false };



    task = qio_task_new(obj, task_callback, &data, NULL);

    src = qio_task_get_source(task);



    qio_task_complete(task);



    g_assert(obj == src);



    object_unref(obj);

    object_unref(src);



    g_assert(data.source == obj);

    g_assert(data.err == NULL);

    g_assert(data.freed == false);

}
