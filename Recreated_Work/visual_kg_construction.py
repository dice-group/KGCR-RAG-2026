import json
import html
import colorsys

from pathlib import Path
from collections import Counter

from pyvis.network import Network

### Completely Optional Representation


def generate_relationship_colors(relationships):
    """
    Generate a unique color for every relationship type.

    Colors are generated dynamically based on the relationship
    types present in the graph.
    """

    relationship_types = sorted({
        relationship["name"]
        for relationship in relationships
    })

    colors = {}

    n = len(relationship_types)

    for i, relationship_type in enumerate(relationship_types):

        # Distribute colors evenly around the HSV color wheel
        hue = i / max(n, 1)

        saturation = 0.70
        value = 0.95

        r, g, b = colorsys.hsv_to_rgb(
            hue,
            saturation,
            value
        )

        color = "#{:02X}{:02X}{:02X}".format(
            int(r * 255),
            int(g * 255),
            int(b * 255)
        )

        colors[relationship_type] = color

    return colors


def read_knowledge_graph(file_path):
    """
    Load entities and relationships from an exported JSON file.

    Returns:
        graph_data: Dictionary containing entities,
                    relationships and summary.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Knowledge graph file not found: {file_path}"
        )

    with file_path.open("r", encoding="utf-8") as file:
        graph_data = json.load(file)

    # Basic validation
    if "entities" not in graph_data:
        raise ValueError("Missing 'entities' in JSON file.")

    if "relationships" not in graph_data:
        raise ValueError("Missing 'relationships' in JSON file.")

    return graph_data


def reconstruct_graph(graph_data):
    """
    Prepare entities and relationships for visualization.

    Automatically adds entities that occur as relationship
    endpoints but are missing from the entity list.
    """

    entities = {}
    relationships = graph_data["relationships"]

    # --------------------------------------------------------
    # Extract entities
    # --------------------------------------------------------

    for entity in graph_data["entities"]:

        key = (entity["label"], entity["name"])

        entities[key] = {
            "id": f"{entity['label']}::{entity['name']}",
            "name": entity["name"],
            "label": entity["label"]
        }

    # --------------------------------------------------------
    # Ensure all relationship endpoints exist
    # --------------------------------------------------------

    for relationship in relationships:

        for endpoint in [
            relationship["startEntity"],
            relationship["endEntity"]
        ]:

            key = (
                endpoint["label"],
                endpoint["name"]
            )

            if key not in entities:

                entities[key] = {
                    "id": f"{endpoint['label']}::{endpoint['name']}",
                    "name": endpoint["name"],
                    "label": endpoint["label"]
                }

    return list(entities.values()), relationships


ENTITY_COLORS = {
    "Person": "#F59E0B",
    "Organization": "#EF4444",
    "Methodology": "#8B5CF6",
    "Model": "#3B82F6",
    "Representation": "#06B6D4",
    "Molecule": "#10B981",
    "Property": "#EC4899",
    "Behavior": "#F97316",
    "Field": "#84CC16",
    "Statistic": "#A78BFA",
    "Finding": "#14B8A6",
    "Sample": "#EAB308",
    "Relationship": "#64748B"
}


def visualize_knowledge_graph(
    entities,
    relationships,
    output_file="visualizations/knowledge_graph.html",
    height="900px",
    width="100%",
    show_edge_labels=True,
    physics=True
):
    """
    Generate an interactive knowledge graph visualization.

    Features:
        - Different colors for entity types
        - Directed relationships
        - Relationship labels
        - Hover information
        - Zooming and dragging
        - Force-directed graph layout
    """

    relationship_colors = generate_relationship_colors(relationships)
    
    output_path = Path(output_file)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Initialize PyVis
    # --------------------------------------------------------

    net = Network(
        height=height,
        width=width,
        directed=True,
        bgcolor="#111827",
        font_color="#F9FAFB",
        notebook=False,
        cdn_resources="in_line"
    )

    # --------------------------------------------------------
    # Helper function
    # --------------------------------------------------------

    def get_node_id(entity):
        return entity["id"]

    # --------------------------------------------------------
    # Add entities as nodes
    # --------------------------------------------------------

    for entity in entities:

        node_id = get_node_id(entity)

        entity_name = entity["name"]
        entity_type = entity["label"]

        color = ENTITY_COLORS.get(
            entity_type,
            "#9CA3AF"
        )

        tooltip = (
            f"<b>{html.escape(entity_name)}</b><br>"
            f"Type: {html.escape(entity_type)}"
        )

        net.add_node(
            node_id,
            label=entity_name,
            title=tooltip,
            color={
                "background": color,
                "border": "#FFFFFF",
                "highlight": {
                    "background": color,
                    "border": "#FFFFFF"
                }
            },
            shape="dot",
            size=25,
            borderWidth=2,
            font={
                "size": 16,
                "color": "#F9FAFB",
                "face": "Arial",
                "bold": True
            }
        )

    # --------------------------------------------------------
    # Add relationships as directed edges
    # --------------------------------------------------------

    for relationship in relationships:

        start = relationship["startEntity"]
        end = relationship["endEntity"]

        source = (
            f"{start['label']}::{start['name']}"
        )

        target = (
            f"{end['label']}::{end['name']}"
        )

        relation_name = relationship["name"]

        edge_label = relation_name.replace("_", " ")

        net.add_edge(
            source,
            target,
            label=edge_label if show_edge_labels else "",
            title=f"Relationship: {html.escape(relation_name)}",
            arrows="to",
            color=relationship_colors[relation_name],
            width=2,
            font={
                "size": 11,
                "color": "#E5E7EB",
                "strokeWidth": 3,
                "strokeColor": "#111827",
                "align": "middle"
            },
            smooth={
                "enabled": True,
                "type": "dynamic"
            }
        )

    # --------------------------------------------------------
    # Configure physics and interaction
    # --------------------------------------------------------

    net.set_options("""
    {
      "physics": {
        "enabled": true,
        "barnesHut": {
          "gravitationalConstant": -6000,
          "centralGravity": 0.25,
          "springLength": 220,
          "springConstant": 0.04,
          "damping": 0.15,
          "avoidOverlap": 0.8
        },
        "minVelocity": 0.75,
        "stabilization": {
          "enabled": true,
          "iterations": 1500,
          "fit": true
        }
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 150,
        "navigationButtons": true,
        "keyboard": {
          "enabled": true
        }
      },
      "edges": {
        "selectionWidth": 3,
        "hoverWidth": 3
      }
    }
    """)

    # --------------------------------------------------------
    # Export visualization
    # --------------------------------------------------------

    # Generate HTML
    html_content = net.generate_html()

    # Generate legend
    legend_html = create_legend_html(
        entities,
        relationships,
        relationship_colors
    )

    # Insert legend into HTML
    html_content = html_content.replace(
        "<body>",
        "<body>" + legend_html,
        1
    )

    # Save HTML using UTF-8 encoding
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
        

    print("\nVisualization successfully generated!")
    print(f"Entities: {len(entities)}")
    print(f"Relationships: {len(relationships)}")
    print(f"Saved to: {output_path.resolve()}")
    f.close()

    return output_path


def create_legend_html(
    entities,
    relationships,
    relationship_colors
):

    entity_types = sorted({
        entity["label"] for entity in entities
    })

    relationship_types = sorted({
        relationship["name"]
        for relationship in relationships
    })

    legend = """
    <div id="graph-legend" style="
        position: fixed;
        top: 15px;
        left: 15px;
        max-height: 85vh;
        overflow-y: auto;
        background: rgba(17, 24, 39, 0.96);
        color: #F9FAFB;
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #374151;
        font-family: Arial, sans-serif;
        font-size: 13px;
        z-index: 9999;
        min-width: 210px;
        max-width: 280px;
    ">

    <h3 style="margin-top:0;">Graph Legend</h3>

    <h4 style="color:#93C5FD;">Entity Types</h4>
    """

    # Entity types
    for entity_type in entity_types:

        color = ENTITY_COLORS.get(
            entity_type,
            "#9CA3AF"
        )

        legend += f"""
        <div style="display:flex; align-items:center; margin-bottom:8px;">
            <span style="
                width:13px;
                height:13px;
                border-radius:50%;
                background:{color};
                margin-right:10px;
                flex-shrink:0;
            "></span>

            <span>{html.escape(entity_type)}</span>
        </div>
        """

    legend += """
    <hr style="border-color:#374151; margin:15px 0;">
    <h4 style="color:#93C5FD;">Relationship Types</h4>
    """

    # Relationship types
    for relationship_type in relationship_types:

        color = relationship_colors[relationship_type]

        display_name = relationship_type.replace("_", " ")

        legend += f"""
        <div style="display:flex; align-items:center; margin-bottom:8px;">
            <span style="
                width:22px;
                height:3px;
                background:{color};
                margin-right:10px;
                flex-shrink:0;
            "></span>

            <span>{html.escape(display_name)}</span>
        </div>
        """

    legend += "</div>"

    return legend


def main():
    INPUT_FILE = "kg_test_results.json"
    OUTPUT_FILE = "visualizations/knowledge_graph.html"


    graph_data = read_knowledge_graph(INPUT_FILE)

    entities, relationships = reconstruct_graph(graph_data)


    print("\nKnowledge Graph Statistics")
    print("=" * 35)

    print(f"Entities: {len(entities)}")
    print(f"Relationships: {len(relationships)}")

    print("\nEntity types:")

    entity_counts = Counter(
        entity["label"] for entity in entities
    )

    for entity_type, count in entity_counts.items():
        print(f"  {entity_type}: {count}")


    visualize_knowledge_graph(
        entities=entities,
        relationships=relationships,
        output_file=OUTPUT_FILE,
        show_edge_labels=True,
        physics=True
    )


if __name__ == "__main__":
    main()
