use crate::components::code_review;
use crate::components::plan_detail;
use crate::components::session_picker;
use crate::components::task_detail;
use crate::components::task_list;

pub enum Screen {
    Chat,
    CodeReview(code_review::State),
    SessionPicker {
        picker: session_picker::State,
        list: task_list::State,
    },
    TaskList(task_list::State),
    TaskDetail {
        detail: task_detail::State,
        list: task_list::State,
    },
    PlanDetail(plan_detail::State),
}
