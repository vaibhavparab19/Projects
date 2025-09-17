import json
import pandas as pd
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from io import BytesIO
import os
from datetime import datetime


class ExportUtils:
    """Utility class for exporting implementation plan results in various formats"""

    def __init__(self):
        self.styles = getSampleStyleSheet()
        self.title_style = ParagraphStyle(
            "CustomTitle",
            parent=self.styles["Heading1"],
            fontSize=18,
            spaceAfter=30,
            textColor=colors.darkblue,
        )
        self.heading_style = ParagraphStyle(
            "CustomHeading",
            parent=self.styles["Heading2"],
            fontSize=14,
            spaceAfter=12,
            textColor=colors.darkgreen,
        )

    def export_to_json(self, data, filename=None):
        """Export data to JSON format"""
        if filename is None:
            filename = (
                f"implementation_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )

        # Create a structured export with metadata
        export_data = {
            "export_metadata": {
                "timestamp": datetime.now().isoformat(),
                "format": "json",
                "version": "1.0",
            },
            "implementation_plan": data,
        }

        buffer = BytesIO()
        json_str = json.dumps(export_data, indent=2, ensure_ascii=False)
        buffer.write(json_str.encode("utf-8"))
        buffer.seek(0)

        return buffer, filename

    def export_to_excel(self, data, filename=None):
        """Export data to Excel format"""
        if filename is None:
            filename = (
                f"implementation_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            )

        buffer = BytesIO()

        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            # Summary sheet
            summary_data = {
                "Export Information": ["Timestamp", "Format", "Version"],
                "Values": [
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Excel",
                    "1.0",
                ],
            }
            pd.DataFrame(summary_data).to_excel(
                writer, sheet_name="Summary", index=False
            )

            # Decision flows sheet
            if "decision_flows" in data:
                flows_data = []
                for segment, flow in data["decision_flows"].items():
                    flows_data.append(
                        {
                            "Segment": segment.replace("_", " ").title(),
                            "Goal": flow.get("goal", ""),
                            "Criteria": flow.get("criteria", ""),
                            "Actions": "; ".join(flow.get("actions", [])),
                            "Context Factors": "; ".join(
                                flow.get("context_factors", [])
                            ),
                            "Success Metrics": "; ".join(
                                flow.get("success_metrics", [])
                            ),
                        }
                    )

                if flows_data:
                    pd.DataFrame(flows_data).to_excel(
                        writer, sheet_name="Decision Flows", index=False
                    )

            # AI Strategy sheet
            if "ai_execution_strategy" in data:
                strategy = data["ai_execution_strategy"]
                strategy_data = [
                    {
                        "Component": "Real-time Decisioning",
                        "Description": strategy.get("real_time_decisioning", ""),
                    },
                    {
                        "Component": "Personalization Engine",
                        "Description": strategy.get("personalization_engine", ""),
                    },
                    {
                        "Component": "Value Prediction",
                        "Description": strategy.get("value_prediction", ""),
                    },
                    {
                        "Component": "Learning System",
                        "Description": strategy.get("learning_system", ""),
                    },
                ]
                pd.DataFrame(strategy_data).to_excel(
                    writer, sheet_name="AI Strategy", index=False
                )

            # Performance Tracking sheet
            if "performance_tracking" in data:
                tracking = data["performance_tracking"]
                tracking_data = []

                if "real_time_metrics" in tracking:
                    for metric in tracking["real_time_metrics"]:
                        tracking_data.append(
                            {"Category": "Real-time Metrics", "Item": metric}
                        )

                tracking_data.extend(
                    [
                        {
                            "Category": "A/B Testing",
                            "Item": tracking.get("ab_testing", ""),
                        },
                        {
                            "Category": "Optimization",
                            "Item": tracking.get("optimization", ""),
                        },
                    ]
                )

                if tracking_data:
                    pd.DataFrame(tracking_data).to_excel(
                        writer, sheet_name="Performance Tracking", index=False
                    )

        buffer.seek(0)
        return buffer, filename

    def export_to_pdf(self, data, filename=None):
        """Export data to PDF format"""
        if filename is None:
            filename = (
                f"implementation_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            )

        buffer = BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18,
        )

        story = []

        # Title
        story.append(
            Paragraph("AI Implementation Plan Report", self.title_style))
        story.append(Spacer(1, 12))

        # Metadata
        story.append(
            Paragraph(
                f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                self.styles["Normal"],
            )
        )
        story.append(Spacer(1, 20))

        # Decision Flows Section
        if "decision_flows" in data:
            story.append(
                Paragraph("Decision Flow Analysis", self.heading_style))

            for segment, flow in data["decision_flows"].items():
                story.append(
                    Paragraph(
                        f"<b>{segment.replace('_', ' ').title()}</b>",
                        self.styles["Heading3"],
                    )
                )

                # Create table for flow details
                flow_data = [
                    ["Goal", flow.get("goal", "N/A")],
                    ["Criteria", flow.get("criteria", "N/A")],
                    ["Actions", "<br/>".join(flow.get("actions", []))],
                    ["Context Factors",
                        "<br/>".join(flow.get("context_factors", []))],
                    ["Success Metrics",
                        "<br/>".join(flow.get("success_metrics", []))],
                ]

                table = Table(flow_data, colWidths=[2 * inch, 4 * inch])
                table.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (0, -1), colors.lightgrey),
                            ("TEXTCOLOR", (0, 0), (-1, -1), colors.black),
                            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                            ("FONTSIZE", (0, 0), (-1, -1), 10),
                            ("GRID", (0, 0), (-1, -1), 1, colors.black),
                            ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ]
                    )
                )

                story.append(table)
                story.append(Spacer(1, 20))

        # AI Execution Strategy Section
        if "ai_execution_strategy" in data:
            story.append(
                Paragraph("AI Execution Strategy", self.heading_style))
            strategy = data["ai_execution_strategy"]

            strategy_data = [
                ["Real-time Decisioning",
                    strategy.get("real_time_decisioning", "N/A")],
                [
                    "Personalization Engine",
                    strategy.get("personalization_engine", "N/A"),
                ],
                ["Value Prediction", strategy.get("value_prediction", "N/A")],
                ["Learning System", strategy.get("learning_system", "N/A")],
            ]

            table = Table(strategy_data, colWidths=[2 * inch, 4 * inch])
            table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (0, -1), colors.lightgreen),
                        ("TEXTCOLOR", (0, 0), (-1, -1), colors.black),
                        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                        ("FONTSIZE", (0, 0), (-1, -1), 10),
                        ("GRID", (0, 0), (-1, -1), 1, colors.black),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ]
                )
            )

            story.append(table)
            story.append(Spacer(1, 20))

        # Performance Tracking Section
        if "performance_tracking" in data:
            story.append(Paragraph("Performance Tracking", self.heading_style))
            tracking = data["performance_tracking"]

            tracking_data = []
            if "real_time_metrics" in tracking:
                tracking_data.append(
                    ["Real-time Metrics",
                        "<br/>".join(tracking["real_time_metrics"])]
                )

            tracking_data.extend(
                [
                    ["A/B Testing", tracking.get("ab_testing", "N/A")],
                    ["Optimization", tracking.get("optimization", "N/A")],
                ]
            )

            table = Table(tracking_data, colWidths=[2 * inch, 4 * inch])
            table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (0, -1), colors.lightyellow),
                        ("TEXTCOLOR", (0, 0), (-1, -1), colors.black),
                        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                        ("FONTSIZE", (0, 0), (-1, -1), 10),
                        ("GRID", (0, 0), (-1, -1), 1, colors.black),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ]
                )
            )

            story.append(table)

        doc.build(story)
        buffer.seek(0)

        return buffer, filename

    def get_sample_data(self):
        """Generate sample implementation plan data for testing"""
        return {
            "decision_flows": {
                "high_value_customers": {
                    "goal": "Maximize revenue from high-value customer segments",
                    "criteria": "Customers with CLV > $5000 and engagement score > 80%",
                    "actions": [
                        "Deploy premium personalized campaigns",
                        "Implement VIP customer journey",
                        "Activate cross-sell recommendations",
                    ],
                    "context_factors": [
                        "Purchase history analysis",
                        "Behavioral engagement patterns",
                        "Seasonal buying trends",
                    ],
                    "success_metrics": [
                        "Revenue per customer increase by 25%",
                        "Campaign engagement rate > 45%",
                        "Customer retention rate > 90%",
                    ],
                }
            },
            "ai_execution_strategy": {
                "real_time_decisioning": "Dynamic customer journey optimization based on real-time behavior analysis",
                "personalization_engine": "Advanced ML algorithms for content and timing personalization",
                "value_prediction": "Predictive analytics for customer lifetime value scoring",
                "learning_system": "Continuous model improvement through feedback loops",
            },
            "performance_tracking": {
                "real_time_metrics": [
                    "Conversion rates by segment",
                    "Customer engagement scores",
                    "Revenue attribution tracking",
                ],
                "ab_testing": "Automated test creation and statistical significance monitoring",
                "optimization": "Continuous model refinement and performance anomaly detection",
            },
        }
