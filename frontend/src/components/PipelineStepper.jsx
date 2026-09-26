import PipelineStage from "./PipelineStage";

export default function PipelineStepper({ stages, selectedIndex, onSelect }) {
  return (
    <div>
      {stages.map((stage, i) => (
        <PipelineStage
          key={stage.name}
          stage={stage}
          index={i}
          isSelected={i === selectedIndex}
          isLast={i === stages.length - 1}
          onSelect={onSelect}
        />
      ))}
    </div>
  );
}
