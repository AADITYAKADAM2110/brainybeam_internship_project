from src.project import build_project


def main() -> None:
    analysis = build_project(force=True)
    print("Project assets generated successfully.")
    print(f"Processed rows: {len(analysis['processed_data'])}")
    print(f"Best model: {analysis['best_model']}")
    print("Reports created:")
    for report_name, report_path in analysis["charts"].items():
        print(f" - {report_name}: {report_path}")


if __name__ == "__main__":
    main()
