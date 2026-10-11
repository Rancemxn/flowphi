import argparse
import sys

from flowphi.runtime import WORKER_MODULES, restore_standard_streams


def main():
    restore_standard_streams()
    if sys.argv[1:2] == ['--run-module']:
        if len(sys.argv) < 3 or sys.argv[2] not in WORKER_MODULES:
            raise SystemExit('Unsupported background job')
        module = sys.argv[2]
        sys.argv = [sys.argv[0], *sys.argv[3:]]
        if module == 'flowphi.ui.worker':
            from flowphi.ui.worker import main as run
        else:
            from flowphi.rendering.pipeline import main as run
        return run()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--chart', help='Chart ZIP/PEZ, JSON or PEC to open')
    parser.add_argument('--project', help='Saved .fingering.json or motion-plan.json')
    parser.add_argument('--demo', action='store_true', help='Open an original example chart and generate its plan')
    args = parser.parse_args()
    from flowphi.ui.appearance import apply_startup_scale
    scale = apply_startup_scale()
    from PySide6.QtCore import QLibraryInfo, QTimer, QTranslator
    from PySide6.QtWidgets import QApplication
    from flowphi.config import OUTPUTS, PROJECTS
    from flowphi.ui.gui import EditorWindow
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    PROJECTS.mkdir(parents=True, exist_ok=True)
    app = QApplication(sys.argv[:1])
    app.setApplicationName('flowphi')
    app.setOrganizationName('flowphi')
    app.setProperty('interface_scale', scale)
    translator = QTranslator(app)
    if translator.load('qtbase_zh_CN', QLibraryInfo.path(QLibraryInfo.TranslationsPath)):
        app.installTranslator(translator)
    window = EditorWindow()
    window.show()
    if args.project:
        QTimer.singleShot(0, lambda: window.open_project(args.project))
    elif args.chart:
        QTimer.singleShot(0, lambda: window.open_chart(args.chart))
    elif args.demo:
        QTimer.singleShot(0, window.open_demo)
    return app.exec()


if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    raise SystemExit(main())
