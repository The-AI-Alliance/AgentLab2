class AscendingNumbersTask:
    desc = "Click on the numbers in ascending order."
    subdomain = "ascending-numbers"


class BisectAngleTask:
    desc = "Find the line that bisects an angle evenly in two."
    subdomain = "bisect-angle"


class BookFlightTask:
    desc = "Search for flight results."
    subdomain = "book-flight"


class BookFlightNodelayTask:
    desc = "[book-flight] Removed animation."
    subdomain = "book-flight-nodelay"


class BuyTicketTask:
    desc = "Buy a ticket that matches the requested criteria."
    subdomain = "buy-ticket"


class ChooseDateTask:
    desc = "Learn to operate a date picker tool."
    subdomain = "choose-date"


class ChooseDateEasyTask:
    desc = "[choose-date] December only."
    subdomain = "choose-date-easy"


class ChooseDateMediumTask:
    desc = "[choose-date] December or November only."
    subdomain = "choose-date-medium"


class ChooseDateNodelayTask:
    desc = "[choose-date] Removed animation."
    subdomain = "choose-date-nodelay"


class ChooseListTask:
    desc = "Choose an item from a drop down list."
    subdomain = "choose-list"


class CircleCenterTask:
    desc = "Find the center of a circle."
    subdomain = "circle-center"


class ClickButtonTask:
    desc = "Click on a specific button in a generated form."
    subdomain = "click-button"


class ClickButtonSequenceTask:
    desc = "Click on buttons in a certain order."
    subdomain = "click-button-sequence"


class ClickCheckboxesTask:
    desc = "Click desired checkboxes."
    subdomain = "click-checkboxes"


class ClickCheckboxesLargeTask:
    desc = "[click-checkboxes] Click at least 5 out of up to 12 checkboxes."
    subdomain = "click-checkboxes-large"


class ClickCheckboxesSoftTask:
    desc = "[click-checkboxes] Paraphrased entries."
    subdomain = "click-checkboxes-soft"


class ClickCheckboxesTransferTask:
    desc = "[click-checkboxes] Train and test on different number of targets."
    subdomain = "click-checkboxes-transfer"


class ClickCollapsibleTask:
    desc = "Click a collapsible element to expand it."
    subdomain = "click-collapsible"


class ClickCollapsible2Task:
    desc = "Find and click on a specified link, from collapsible elements."
    subdomain = "click-collapsible-2"


class ClickCollapsible2NodelayTask:
    desc = "[click-collapsible-2] Removed animation."
    subdomain = "click-collapsible-2-nodelay"


class ClickCollapsibleNodelayTask:
    desc = "[click-collapsible] Removed animation."
    subdomain = "click-collapsible-nodelay"


class ClickColorTask:
    desc = "Click the specified color."
    subdomain = "click-color"


class ClickDialogTask:
    desc = "Click the button to close the dialog box."
    subdomain = "click-dialog"


class ClickDialog2Task:
    desc = "Click a specific button in a dialog box."
    subdomain = "click-dialog-2"


class ClickLinkTask:
    desc = "Click on a specified link in text."
    subdomain = "click-link"


class ClickMenuTask:
    desc = "Click menu items."
    subdomain = "click-menu"


class ClickMenu2Task:
    desc = "Find a specific item from a menu."
    subdomain = "click-menu-2"


class ClickOptionTask:
    desc = "Click option boxes."
    subdomain = "click-option"


class ClickPieTask:
    desc = "Click items on a pie menu."
    subdomain = "click-pie"
    nondeterministic = True


class ClickPieNodelayTask:
    desc = "[click-pie] Removed animation."
    subdomain = "click-pie-nodelay"
    nondeterministic = True


class ClickScrollListTask:
    desc = "Click multiple items from a scroll list."
    subdomain = "click-scroll-list"


class ClickShadesTask:
    desc = "Click the shades that match a specified color."
    subdomain = "click-shades"


class ClickShapeTask:
    desc = "Click on a specific shape."
    subdomain = "click-shape"


class ClickTabTask:
    desc = "Click on a tab element."
    subdomain = "click-tab"


class ClickTab2Task:
    desc = "Click a link inside a specific tab element."
    subdomain = "click-tab-2"


class ClickTab2EasyTask:
    desc = "[click-tab-2] One 1 tab."
    subdomain = "click-tab-2-easy"


class ClickTab2HardTask:
    desc = "[click-tab-2] Varying number of tabs from 2 to 6."
    subdomain = "click-tab-2-hard"


class ClickTab2MediumTask:
    desc = "[click-tab-2] Choose between a link or ‘no match’."
    subdomain = "click-tab-2-medium"


class ClickTestTask:
    desc = "Click on a single button."
    subdomain = "click-test"


class ClickTest2Task:
    desc = "Click on one of two buttons."
    subdomain = "click-test-2"


class ClickTestTransferTask:
    desc = "[click-test] Different buttons during train and test."
    subdomain = "click-test-transfer"


class ClickWidgetTask:
    desc = "Click on a specific widget in a generated form."
    subdomain = "click-widget"


class CopyPasteTask:
    desc = "Copy text and paste it into an input."
    subdomain = "copy-paste"


class CopyPaste2Task:
    desc = "Copy text from a specific textarea and paste it into an input."
    subdomain = "copy-paste-2"


class CountShapeTask:
    desc = "Count number of shapes."
    subdomain = "count-shape"


class CountSidesTask:
    desc = "Count the number of sides on a shape."
    subdomain = "count-sides"


class DailyCalendarTask:
    desc = "Create an event on a daily calendar."
    subdomain = "daily-calendar"


class DragBoxTask:
    desc = "Drag the smaller box into the larger box."
    subdomain = "drag-box"


class DragCircleTask:
    desc = "Drag an item in a specified direction."
    subdomain = "drag-circle"


class DragCubeTask:
    desc = "Drag a 3D cube to show a specific face."
    subdomain = "drag-cube"


class DragItemsTask:
    desc = "Drag items in a list, in a specified direction"
    subdomain = "drag-items"


class DragItemsGridTask:
    desc = "Drag items in a 2D grid around."
    subdomain = "drag-items-grid"


class DragShapesTask:
    desc = "Drag shapes into a box."
    subdomain = "drag-shapes"


class DragShapes2Task:
    desc = "Drag shapes into boxes, categorized by type."
    subdomain = "drag-shapes-2"


class DragSingleShapeTask:
    desc = "Drag a randomly generated shape in a specified direction."
    subdomain = "drag-single-shape"


class DragSortNumbersTask:
    desc = "Drag numbers into sorted ascending order."
    subdomain = "drag-sort-numbers"


class DrawCircleTask:
    desc = "Draw a circle around a marked point."
    subdomain = "draw-circle"


class DrawLineTask:
    desc = "Draw a line through a marked point."
    subdomain = "draw-line"


class EmailInboxTask:
    desc = "Navigate through an email inbox and perform some actions."
    subdomain = "email-inbox"


class EmailInboxDeleteTask:
    desc = "[email-inbox] No scrolling + 1 subtask."
    subdomain = "email-inbox-delete"


class EmailInboxForwardTask:
    desc = "[email-inbox] No scrolling + 1 subtask."
    subdomain = "email-inbox-forward"


class EmailInboxForwardNlTask:
    desc = "[email-inbox-forward] varied instruction texts (30 templates)."
    subdomain = "email-inbox-forward-nl"


class EmailInboxForwardNlTurkTask:
    desc = "[email-inbox-forward] varied instruction texts (100 templates)."
    subdomain = "email-inbox-forward-nl-turk"


class EmailInboxImportantTask:
    desc = "[email-inbox] No scrolling + 1 subtask."
    subdomain = "email-inbox-important"


class EmailInboxNlTurkTask:
    desc = "[email-inbox] varied instruction texts (100 templates for each subtask)."
    subdomain = "email-inbox-nl-turk"


class EmailInboxNoscrollTask:
    desc = "[email-inbox] No scrolling."
    subdomain = "email-inbox-noscroll"


class EmailInboxReplyTask:
    desc = "[email-inbox] No scrolling + 1 subtask."
    subdomain = "email-inbox-reply"


class EmailInboxStarReplyTask:
    desc = "[email-inbox] No scrolling + 2 subtasks."
    subdomain = "email-inbox-star-reply"


class EnterDateTask:
    desc = "Use the date input to pick the correct date."
    subdomain = "enter-date"


class EnterPasswordTask:
    desc = "Enter the password into the form."
    subdomain = "enter-password"


class EnterTextTask:
    desc = "Enter given text to a textfield."
    subdomain = "enter-text"


class EnterText2Task:
    desc = "Convert given text to upper or lower case."
    subdomain = "enter-text-2"


class EnterTextDynamicTask:
    desc = "Enter dynamically generated text to a textfield."
    subdomain = "enter-text-dynamic"


class EnterTimeTask:
    desc = "Enter the specified time into the input."
    subdomain = "enter-time"


class FindGreatestTask:
    desc = "Find the card with the greatest number."
    subdomain = "find-greatest"


class FindMidpointTask:
    desc = "Find the shortest mid-point of two points."
    subdomain = "find-midpoint"


class FindWordTask:
    desc = "Find nth word in a block of text."
    subdomain = "find-word"


class FocusTextTask:
    desc = "Focus into a text input."
    subdomain = "focus-text"


class FocusText2Task:
    desc = "Focus on a specific text input."
    subdomain = "focus-text-2"


class FormSequenceTask:
    desc = "Perform a series of instructions on a form."
    subdomain = "form-sequence"


class FormSequence2Task:
    desc = "Perform a series of instructions on a form."
    subdomain = "form-sequence-2"


class FormSequence3Task:
    desc = "Perform a series of instructions on a form."
    subdomain = "form-sequence-3"


class GenerateNumberTask:
    desc = "Generate a random number that meets certain criteria."
    subdomain = "generate-number"


class GridCoordinateTask:
    desc = "Find the Cartesian coordinates on a grid."
    subdomain = "grid-coordinate"


class GuessNumberTask:
    desc = "Guess the number."
    subdomain = "guess-number"


class HighlightTextTask:
    desc = "Highlight all the text."
    subdomain = "highlight-text"


class HighlightText2Task:
    desc = "Highlight the specified paragraph."
    subdomain = "highlight-text-2"


class HotColdTask:
    desc = "Find and click on the hot area."
    subdomain = "hot-cold"


class IdentifyShapeTask:
    desc = "Identify a randomly generated shape."
    subdomain = "identify-shape"


class LoginUserTask:
    desc = "Enter user login details into the form."
    subdomain = "login-user"


class LoginUserPopupTask:
    desc = "[login-user] Random popup."
    subdomain = "login-user-popup"


class MultiLayoutsTask:
    desc = "Fill in forms of varying layouts."
    subdomain = "multi-layouts"


class MultiOrderingsTask:
    desc = "Fill in forms with shuffled field orderings."
    subdomain = "multi-orderings"


class NavigateTreeTask:
    desc = "Navigate a file tree to find a specified file or folder."
    subdomain = "navigate-tree"


class NumberCheckboxesTask:
    desc = "Draw a given number using checkboxes."
    subdomain = "number-checkboxes"


class OddOrEvenTask:
    desc = "Mark each number as odd or even."
    subdomain = "odd-or-even"


class OrderFoodTask:
    desc = "Order food items from a menu."
    subdomain = "order-food"


class PhoneBookTask:
    desc = "Find a contact in a phone book."
    subdomain = "phone-book"


class ReadTableTask:
    desc = "Read information out from a table."
    subdomain = "read-table"


class ReadTable2Task:
    desc = "Read multiple pieces of information out from a table."
    subdomain = "read-table-2"


class ResizeTextareaTask:
    desc = "Resize a textarea in a given direction."
    subdomain = "resize-textarea"


class RightAngleTask:
    desc = "Given two points, add a third point to create a right angle."
    subdomain = "right-angle"


class ScrollTextTask:
    desc = "Scroll through a text area element and enter last word into text area."
    subdomain = "scroll-text"


class ScrollText2Task:
    desc = "Scroll through a text area in a given direction."
    subdomain = "scroll-text-2"


class SearchEngineTask:
    desc = "Search through a bunch of results to find a specified link."
    subdomain = "search-engine"


class SignAgreementTask:
    desc = "Sign a user agreement."
    subdomain = "sign-agreement"


class SimpleAlgebraTask:
    desc = "Solve for X."
    subdomain = "simple-algebra"


class SimpleArithmeticTask:
    desc = "Perform some arithmetic math operations."
    subdomain = "simple-arithmetic"


class SocialMediaTask:
    desc = "Interact with a social media feed."
    subdomain = "social-media"


class SocialMediaAllTask:
    desc = "[social-media] Do some action on all matching entries."
    subdomain = "social-media-all"


class SocialMediaSomeTask:
    desc = "[social-media] Do some action on some matching entries."
    subdomain = "social-media-some"


class StockMarketTask:
    desc = "Buy from the stock market below a specified price."
    subdomain = "stock-market"


class TerminalTask:
    desc = "Use the terminal to delete a file."
    subdomain = "terminal"
    nondeterministic = True


class TextEditorTask:
    desc = "Modify a text's style in a text-editor."
    subdomain = "text-editor"


class TextTransformTask:
    desc = "Enter slightly transformed text into a text box."
    subdomain = "text-transform"


class TicTacToeTask:
    desc = "Win a game of tic-tac-toe."
    subdomain = "tic-tac-toe"


class UnicodeTestTask:
    desc = "Click on the button with the correct Unicode text."
    subdomain = "unicode-test"


class UseAutocompleteTask:
    desc = "Use autocomplete element efficiently."
    subdomain = "use-autocomplete"


class UseAutocompleteNodelayTask:
    desc = "[use-autocomplete] Removed delay."
    subdomain = "use-autocomplete-nodelay"


class UseColorwheelTask:
    desc = "Use a color wheel."
    subdomain = "use-colorwheel"


class UseColorwheel2Task:
    desc = "Use a color wheel given specific random color."
    subdomain = "use-colorwheel-2"


class UseSliderTask:
    desc = "Use a slider to select a particular value."
    subdomain = "use-slider"


class UseSlider2Task:
    desc = "Use sliders to create a given combination."
    subdomain = "use-slider-2"


class UseSpinnerTask:
    desc = "Use a spinner to select given number."
    subdomain = "use-spinner"


class VisualAdditionTask:
    desc = "Count the total number of blocks."
    subdomain = "visual-addition"
    nondeterministic = True


ALL_MINIWOB_TASKS = [
    AscendingNumbersTask,
    BisectAngleTask,
    BookFlightTask,
    BookFlightNodelayTask,
    BuyTicketTask,
    ChooseDateTask,
    ChooseDateEasyTask,
    ChooseDateMediumTask,
    ChooseDateNodelayTask,
    ChooseListTask,
    CircleCenterTask,
    ClickButtonTask,
    ClickButtonSequenceTask,
    ClickCheckboxesTask,
    ClickCheckboxesLargeTask,
    ClickCheckboxesSoftTask,
    ClickCheckboxesTransferTask,
    ClickCollapsibleTask,
    ClickCollapsible2Task,
    ClickCollapsible2NodelayTask,
    ClickCollapsibleNodelayTask,
    ClickColorTask,
    ClickDialogTask,
    ClickDialog2Task,
    ClickLinkTask,
    ClickMenuTask,
    ClickMenu2Task,
    ClickOptionTask,
    ClickPieTask,
    ClickPieNodelayTask,
    ClickScrollListTask,
    ClickShadesTask,
    ClickShapeTask,
    ClickTabTask,
    ClickTab2Task,
    ClickTab2EasyTask,
    ClickTab2HardTask,
    ClickTab2MediumTask,
    ClickTestTask,
    ClickTest2Task,
    ClickTestTransferTask,
    ClickWidgetTask,
    CopyPasteTask,
    CopyPaste2Task,
    CountShapeTask,
    CountSidesTask,
    DailyCalendarTask,
    DragBoxTask,
    DragCircleTask,
    DragCubeTask,
    DragItemsTask,
    DragItemsGridTask,
    DragShapesTask,
    DragShapes2Task,
    DragSingleShapeTask,
    DragSortNumbersTask,
    DrawCircleTask,
    DrawLineTask,
    EmailInboxTask,
    EmailInboxDeleteTask,
    EmailInboxForwardTask,
    EmailInboxForwardNlTask,
    EmailInboxForwardNlTurkTask,
    EmailInboxImportantTask,
    EmailInboxNlTurkTask,
    EmailInboxNoscrollTask,
    EmailInboxReplyTask,
    EmailInboxStarReplyTask,
    EnterDateTask,
    EnterPasswordTask,
    EnterTextTask,
    EnterText2Task,
    EnterTextDynamicTask,
    EnterTimeTask,
    FindGreatestTask,
    FindMidpointTask,
    FindWordTask,
    FocusTextTask,
    FocusText2Task,
    FormSequenceTask,
    FormSequence2Task,
    FormSequence3Task,
    GenerateNumberTask,
    GridCoordinateTask,
    GuessNumberTask,
    HighlightTextTask,
    HighlightText2Task,
    HotColdTask,
    IdentifyShapeTask,
    LoginUserTask,
    LoginUserPopupTask,
    MultiLayoutsTask,
    MultiOrderingsTask,
    NavigateTreeTask,
    NumberCheckboxesTask,
    OddOrEvenTask,
    OrderFoodTask,
    PhoneBookTask,
    ReadTableTask,
    ReadTable2Task,
    ResizeTextareaTask,
    RightAngleTask,
    ScrollTextTask,
    ScrollText2Task,
    SearchEngineTask,
    SignAgreementTask,
    SimpleAlgebraTask,
    SimpleArithmeticTask,
    SocialMediaTask,
    SocialMediaAllTask,
    SocialMediaSomeTask,
    StockMarketTask,
    TerminalTask,
    TextEditorTask,
    TextTransformTask,
    TicTacToeTask,
    UnicodeTestTask,
    UseAutocompleteTask,
    UseAutocompleteNodelayTask,
    UseColorwheelTask,
    UseColorwheel2Task,
    UseSliderTask,
    UseSlider2Task,
    UseSpinnerTask,
    VisualAdditionTask,
]
