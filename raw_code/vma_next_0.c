static struct vm_area_struct *vma_next(struct vm_area_struct *vma)

{

    return (TAILQ_NEXT(vma, vma_link));

}
