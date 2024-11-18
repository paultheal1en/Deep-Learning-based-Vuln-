Object *qio_task_get_source(QIOTask *task)

{

    object_ref(task->source);

    return task->source;

}
